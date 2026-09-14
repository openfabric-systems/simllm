"""Run the frozen PLACE-2 unique-nic GOAL-rank mapping study.

Cells U1 (identity), U2 (permutation), U4 (refusals) and U5 (wire identity)
follow ``expectations.md``; cell U3 (shared endpoints) follows the replacement
path of ``expectations-amendment-2026-09-13.md``. The seven compatibility
digests and the six m5 check-B makespans are fatal, unscored guards.

Run from the repository root with the local environment sourced, so the
``htsim_rnic`` and ``txt2bin`` binaries resolve::

    PYTHONPATH=. python examples/unique_nic_mapping_v1/run_study.py
    PYTHONPATH=. python examples/unique_nic_mapping_v1/run_study.py --check

Artifacts go to a fresh directory under ``--output-root``, which defaults to
``$SIMLLM_DATA_ROOT/unique_nic_mapping_v1``. A run writes the tracked
``results.json``; ``--check`` reruns the study and requires the tracked
status, findings, evidence and observation to reproduce exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-13.json"
TRACKED_RESULTS = STUDY_DIR / "results.json"
EXPECTATIONS_COMMIT = "d348c3261851dc4b33be94e65749072effbf87cd"
AMENDMENT_COMMIT = "2e56b337c0a6c041769b7916efdafa62a23f0050"
RESULT_SCHEMA = "simllm-unique-nic-mapping-result-v1"
DATA_ROOT_ENV = "SIMLLM_DATA_ROOT"
STUDY_FILES = frozenset(
    {
        "examples/unique_nic_mapping_v1/run_study.py",
        "examples/unique_nic_mapping_v1/results.json",
    }
)
PROFILE = "rnic-nn-fluid"
LINKSPEED_BPS = 400_000_000_000
U3_TAG = 1_000
U3_WORLD = 16
REPRODUCED_KEYS = ("status", "findings", "evidence", "observation")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _is_ancestor(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return completed.returncode == 0


def _require_provenance() -> None:
    """Refuse a run whose code or frozen expectations cannot be named."""

    for commit in (EXPECTATIONS_COMMIT, AMENDMENT_COMMIT):
        if not _is_ancestor(commit):
            raise SystemExit(f"expectations commit {commit} is not an ancestor of HEAD")
    changed = [
        line[3:]
        for line in _git("status", "--porcelain", "--untracked-files=all").splitlines()
    ]
    unexpected = sorted(path for path in changed if path not in STUDY_FILES)
    if unexpected:
        raise SystemExit(
            "the study requires a clean worktree apart from its own harness and "
            "results: " + ", ".join(unexpected)
        )
    import simllm

    if not Path(simllm.__file__).resolve().is_relative_to(REPOSITORY_ROOT):
        raise SystemExit(
            "simllm resolves outside this checkout; run from the repository root "
            "with PYTHONPATH=."
        )


def _m5_study() -> Any:
    """Load the m5 runner for its declared geometry and step shapes."""

    path = REPOSITORY_ROOT / "examples" / "m5" / "run_m5.py"
    spec = importlib.util.spec_from_file_location("simllm_m5_study", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load examples/m5/run_m5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_sink(
    workdir: Path,
    *,
    dims: Any,
    ep_ranks: range,
    record: Any,
    placement: Any = None,
    fabric: Any = None,
) -> tuple[Any, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=PROFILE,
            tp_ranks=(0,),
            dims=dims,
            workdir=workdir,
            ep_ranks=tuple(ep_ranks),
            linkspeed_bps=LINKSPEED_BPS,
            placement_manifest=placement,
            goal_rank_mapping="gpu-rank" if fabric is None else "unique-nic",
            fabric_manifest=fabric,
        )
    )
    result = sink(record)
    if result is None:
        raise RuntimeError("the study step produced no network work")
    return sink, result


def _goal_files(workdir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(workdir.glob("*.goal"))}


def _completion_files(workdir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(workdir.glob(f"*.{PROFILE}.csv"))
    }


def _backend_times(workdir: Path) -> tuple[dict[tuple[int, int, int], tuple[int, int]], int, bool]:
    """Completion times keyed by ``(goal source, goal destination, tag)``.

    Under ``gpu-rank`` GOAL ranks and tags are the semantic ones, so the keys
    are semantic; a repeated key is reported rather than overwritten silently.
    """

    from simllm.backends.htsim_rnic import parse_completion_csv

    times: dict[tuple[int, int, int], tuple[int, int]] = {}
    rows = 0
    unique = True
    for path in sorted(workdir.glob(f"*.{PROFILE}.csv")):
        for flow in parse_completion_csv(path):
            rows += 1
            key = (flow.source, flow.destination, flow.tag)
            unique = unique and key not in times
            times[key] = (flow.start_time_ps, flow.completion_time_ps)
    return times, rows, unique


def _joined_times(sink: Any) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Joined unique-nic completion times keyed by the semantic segment."""

    return {
        (item.source_rank, item.destination_rank, item.tag): (
            item.start_time_ps,
            item.completion_time_ps,
        )
        for item in sink.fabric_join_outcomes[0].segments
    }


def _canonical_goal(text: str, goal_to_semantic: dict[int, int]) -> dict[int, list[str]]:
    """Rank blocks keyed by semantic rank, peer ranks mapped back, labels dropped."""

    blocks: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines()[1:]:
        if line.startswith("rank "):
            current = goal_to_semantic[int(line.split()[1])]
            blocks[current] = []
        elif line != "}":
            words = line.partition(": ")[2].split()
            if words and words[0] in ("send", "recv"):
                words[3] = str(goal_to_semantic[int(words[3])])
            blocks[current].append(" ".join(words))
    return blocks


def _reversed_nic_fabric(placement: Any) -> Any:
    """U2 inventory: one NIC per GPU, NIC ``j`` affine to local rank ``3 - j``."""

    from simllm.placement import (
        FabricNodePlacement,
        FabricTopologyManifest,
        GpuFabricPlacement,
        NicFabricPlacement,
    )

    hosts: list[str] = []
    for rank in sorted(placement.ranks, key=lambda item: item.global_rank):
        if rank.hostname not in hosts:
            hosts.append(rank.hostname)
    nodes = []
    for host in hosts:
        ranks = sorted(
            (rank for rank in placement.ranks if rank.hostname == host),
            key=lambda item: item.local_rank,
        )
        width = len(ranks)
        nic_ids = [f"{host}:nic-{index}" for index in range(width)]
        nodes.append(
            FabricNodePlacement(
                node_id=host,
                pool_role="declared",
                gpus=tuple(
                    GpuFabricPlacement(
                        global_rank=rank.global_rank,
                        gpu_id=f"sim-gpu-{rank.global_rank:04d}",
                        node_id=host,
                        pcie_location=f"{host}/pcie-{rank.local_rank}",
                        nic_id=nic_ids[width - 1 - rank.local_rank],
                    )
                    for rank in ranks
                ),
                nics=tuple(
                    NicFabricPlacement(
                        nic_id=nic_ids[index],
                        node_id=host,
                        fabric_location=f"{host}/nic-{index}",
                        affine_gpu_rank=ranks[width - 1 - index].global_rank,
                    )
                    for index in range(width)
                ),
            )
        )
    return FabricTopologyManifest(nodes=nodes, goal_rank_mapping="unique-nic")


def _run_u1(output_root: Path, m5: Any) -> dict[str, Any]:
    from simllm.placement import RankMapper, disaggregated_manifests

    manifests = disaggregated_manifests(prefill_nodes=1, decode_nodes=1)
    placement = manifests.placement
    fabric = replace(manifests.fabric, goal_rank_mapping="unique-nic")
    mapper = RankMapper(placement, mode="unique-nic", fabric=fabric)
    shapes: dict[str, Any] = {}
    for shape, build in m5.STEP_SHAPES.items():
        runs = {}
        for mode, selected_fabric in (("gpu-rank", None), ("unique-nic", fabric)):
            workdir = output_root / "u1" / shape / mode
            runs[mode] = (
                *_run_sink(
                    workdir,
                    dims=m5.moe_dims(U3_WORLD),
                    ep_ranks=range(U3_WORLD),
                    record=build(),
                    placement=placement,
                    fabric=selected_fabric,
                ),
                _goal_files(workdir),
                workdir,
            )
        gpu_sink, gpu_result, gpu_files, gpu_workdir = runs["gpu-rank"]
        nic_sink, nic_result, nic_files, nic_workdir = runs["unique-nic"]
        nic_join = nic_sink.fabric_join_outcomes[0]
        gpu_times, gpu_rows, gpu_keys_unique = _backend_times(gpu_workdir)
        nic_times = _joined_times(nic_sink)
        shapes[shape] = {
            "completion_files_identical": bool(gpu_files)
            and _completion_files(gpu_workdir) == _completion_files(nic_workdir),
            "goal_artifacts": len(nic_files),
            "goal_text_identical": bool(gpu_files) and gpu_files == nic_files,
            "gpu_rank_backend_rows": gpu_rows,
            "gpu_rank_keys_unique": gpu_keys_unique,
            "joined_endpoints_identity": all(
                (item.goal_source_rank, item.goal_destination_rank)
                == (item.source_rank, item.destination_rank)
                for item in nic_join.segments
            ),
            "joined_rows": len(nic_join.segments),
            "joined_times_identical": len(nic_times) == len(nic_join.segments)
            and nic_times == gpu_times,
            "makespan_ps": nic_result.step_latency_ps,
            "num_flows": nic_sink.outcomes[0].num_flows,
            "outcomes_identical": (
                gpu_sink.outcomes == nic_sink.outcomes
                and gpu_sink.locality_outcomes == nic_sink.locality_outcomes
            ),
            "step_result_identical": gpu_result == nic_result,
            "tag_multiplier": nic_join.tag_multiplier,
        }
    return {
        "goal_rank_identity": all(mapper.goal_rank(rank) == rank for rank in range(16)),
        "nic_count": mapper.num_goal_ranks(),
        "rank_count": len(placement.ranks),
        "shapes": shapes,
    }


def _run_u2(output_root: Path, m5: Any) -> dict[str, Any]:
    from simllm.placement import RankMapper, declared_manifest

    placement = declared_manifest(tp=1, dp=8, gpus_per_node=4)
    fabric = _reversed_nic_fabric(placement)
    mapper = RankMapper(placement, mode="unique-nic", fabric=fabric)
    formula_holds = all(
        mapper.goal_rank(rank.global_rank)
        == 4 * int(rank.hostname.removeprefix("node-")) + (3 - rank.local_rank)
        for rank in placement.ranks
    )
    inverse = {mapper.goal_rank(rank): rank for rank in range(8)}
    identity = {rank: rank for rank in range(8)}
    shapes: dict[str, Any] = {}
    for shape, build in m5.STEP_SHAPES.items():
        runs = {}
        for mode, selected_fabric in (("gpu-rank", None), ("unique-nic", fabric)):
            workdir = output_root / "u2" / shape / mode
            runs[mode] = (
                *_run_sink(
                    workdir,
                    dims=m5.moe_dims(8),
                    ep_ranks=range(8),
                    record=build(),
                    placement=placement,
                    fabric=selected_fabric,
                ),
                _goal_files(workdir),
                workdir,
            )
        gpu_sink, gpu_result, gpu_files, gpu_workdir = runs["gpu-rank"]
        nic_sink, nic_result, nic_files, _nic_workdir = runs["unique-nic"]
        gpu_times, gpu_rows, gpu_keys_unique = _backend_times(gpu_workdir)
        nic_segments = nic_sink.fabric_join_outcomes[0].segments
        nic_times = _joined_times(nic_sink)
        shapes[shape] = {
            "goal_artifacts": len(nic_files),
            "goal_names_identical": sorted(gpu_files) == sorted(nic_files),
            "goal_text_differs": any(
                gpu_files[name] != nic_files.get(name) for name in gpu_files
            ),
            "goal_text_differs_only_in_rank_numbers": bool(gpu_files)
            and sorted(gpu_files) == sorted(nic_files)
            and all(
                _canonical_goal(nic_files[name].decode(), inverse)
                == _canonical_goal(gpu_files[name].decode(), identity)
                for name in gpu_files
            ),
            "joined_endpoints_permuted": all(
                (item.goal_source_rank, item.goal_destination_rank)
                == (mapper.goal_rank(item.source_rank), mapper.goal_rank(item.destination_rank))
                for item in nic_segments
            ),
            "joined_rows": len(nic_segments),
            "gpu_rank_backend_rows": gpu_rows,
            "gpu_rank_keys_unique": gpu_keys_unique,
            "joined_times_identical": bool(nic_segments)
            and len(nic_times) == len(nic_segments)
            and nic_times == gpu_times,
            "makespan_ps": nic_result.step_latency_ps,
            "num_flows": nic_sink.outcomes[0].num_flows,
            "outcomes_identical": (
                gpu_sink.outcomes == nic_sink.outcomes
                and gpu_sink.locality_outcomes == nic_sink.locality_outcomes
            ),
            "step_result_identical": gpu_result == nic_result,
        }
    return {"goal_rank_formula_holds": formula_holds, "shapes": shapes}


def _run_u3(output_root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic
    from simllm.goal import to_binary
    from simllm.placement import RankMapper, declared_manifest, declared_shared_nic_fabric
    from simllm.traffic import (
        CollectiveCommunicationPhase,
        DirectedCollectiveSegment,
        classify_step_locality,
        plan_fabric_goal_projection,
        render_fabric_phase_goal,
    )

    cell = frozen["cells"]["u3_shared"]
    placement = declared_manifest(tp=1, dp=U3_WORLD)
    ranks = tuple(range(U3_WORLD))
    cases: dict[str, Any] = {}
    for payload_bytes in cell["payload_bytes"]:
        mappers = [("gpu-rank", 1, RankMapper(placement))]
        mappers.extend(
            (
                "unique-nic",
                gpus_per_nic,
                RankMapper(
                    placement,
                    mode="unique-nic",
                    fabric=declared_shared_nic_fabric(placement, gpus_per_nic=gpus_per_nic),
                ),
            )
            for gpus_per_nic in cell["gpus_per_nic"]
        )
        for mode, gpus_per_nic, mapper in mappers:
            label = f"s{payload_bytes}-{mode}-g{gpus_per_nic}"
            workdir = output_root / "u3" / label
            workdir.mkdir(parents=True)
            phase = CollectiveCommunicationPhase(
                phase_id="u3:all-pairs",
                layer=0,
                participants=ranks,
                segments=tuple(
                    DirectedCollectiveSegment(source, destination, payload_bytes, U3_TAG)
                    for source in ranks
                    for destination in ranks
                    if source != destination
                ),
                operation_id="u3:all-pairs",
            )
            plan = classify_step_locality((phase,), rank_mapper=mapper)
            projection = plan_fabric_goal_projection(plan.phases, rank_mapper=mapper)
            trace = render_fabric_phase_goal(
                plan.phases[0], rank_mapper=mapper, projection=projection
            )
            goal_path = trace.write(workdir / f"{label}.goal")
            run = run_htsim_rnic(
                HtsimRnicConfig(
                    goal_bin=to_binary(goal_path),
                    profile=PROFILE,
                    linkspeed_bps=LINKSPEED_BPS,
                    completion_csv=workdir / f"{label}.{PROFILE}.csv",
                )
            )
            joined = projection.join_completions(run.flows)
            egress = Counter(message.source_rank for message in trace.messages)
            ingress = Counter(message.destination_rank for message in trace.messages)
            cases[label] = {
                "backend_rows": len(run.flows),
                "completion_rows_sha256": _sha256_bytes(
                    _json_bytes([asdict(flow) for flow in run.flows])
                ),
                "fabric_bytes": plan.fabric_bytes,
                "fabric_messages": len(trace.messages),
                "flow_fct_ps": sorted({flow.fct_ps for flow, _ in joined}),
                "flow_start_ps": sorted({flow.start_time_ps for flow, _ in joined}),
                "flows_per_endpoint_egress": sorted(set(egress.values())),
                "flows_per_endpoint_ingress": sorted(set(ingress.values())),
                "goal_ranks": trace.num_ranks,
                "goal_sha256": _sha256_bytes(goal_path.read_bytes()),
                "gpus_per_nic": gpus_per_nic,
                "joined_rows": len(joined),
                "keys_unique_per_endpoint_pair": len(
                    {
                        (message.source_rank, message.destination_rank, message.tag)
                        for message in trace.messages
                    }
                )
                == len(trace.messages),
                "makespan_ps": run.job_completion_time_ps(),
                "mode": mode,
                "nvlink_bytes": plan.nvlink_bytes,
                "payload_bytes": payload_bytes,
                "quiescent": run.quiescent,
                "tag_multiplier": projection.tag_multiplier,
            }
    return {"cases": cases}


def _refusal(label: str, action: Any, workdir: Path | None = None) -> dict[str, Any]:
    try:
        action()
    except (TypeError, ValueError) as error:
        refused, diagnostic = True, f"{type(error).__name__}: {error}"
    else:
        refused, diagnostic = False, ""
    return {
        "diagnostic": diagnostic,
        "label": label,
        "refused": refused,
        "workdir_absent": workdir is None or not workdir.exists(),
    }


def _run_u4(output_root: Path, m5: Any) -> dict[str, Any]:
    from simllm.backends import HtsimStepSinkConfig
    from simllm.placement import RankMapper, declared_manifest, declared_shared_nic_fabric
    from simllm.traffic import (
        DirectedCollectiveSegment,
        FabricGoalProjection,
        FabricSegmentProjection,
    )

    placement = declared_manifest(tp=1, dp=U3_WORLD)
    fabric = declared_shared_nic_fabric(placement, gpus_per_nic=2)
    node = fabric.nodes[0]
    missing_nic = replace(
        fabric,
        nodes=[
            replace(node, gpus=(replace(node.gpus[0], nic_id="node-0:nic-missing"), *node.gpus[1:])),
            *fabric.nodes[1:],
        ],
    )
    other_placement_fabric = declared_shared_nic_fabric(
        declared_manifest(tp=1, dp=8), gpus_per_nic=2
    )

    def sink_config(label: str, **override: Any) -> tuple[Any, Path]:
        workdir = output_root / "u4" / label
        values = {
            "profile": PROFILE,
            "tp_ranks": (0,),
            "dims": m5.moe_dims(U3_WORLD),
            "workdir": workdir,
            "ep_ranks": tuple(range(U3_WORLD)),
            "placement_manifest": placement,
            "goal_rank_mapping": "unique-nic",
            "fabric_manifest": fabric,
        }
        values.update(override)
        return (lambda: HtsimStepSinkConfig(**values)), workdir

    segment = DirectedCollectiveSegment(0, 8, 64, U3_TAG)
    row = FabricSegmentProjection(0, "u4:duplicate", 0, 4, U3_TAG, segment)
    refusals = [
        _refusal(
            "unique-nic without fabric",
            *sink_config("without-fabric", fabric_manifest=None),
        ),
        _refusal(
            "fabric missing a NIC for one GPU",
            lambda: RankMapper(placement, mode="unique-nic", fabric=missing_nic),
        ),
        _refusal(
            "fabric GPU set differs from placement",
            lambda: RankMapper(placement, mode="unique-nic", fabric=other_placement_fabric),
        ),
        _refusal(
            "gpus_per_nic does not divide node width",
            lambda: declared_shared_nic_fabric(placement, gpus_per_nic=3),
        ),
        _refusal("unique-nic with peer_packet", *sink_config("peer-packet", peer_packet=object())),
        _refusal(
            "unique-nic with topology file",
            *sink_config("topology", topology=Path("declared-clos.topo")),
        ),
        _refusal(
            "unique-nic with flow_session",
            *sink_config("flow-session", flow_session=object()),
        ),
        _refusal(
            "unique-nic with dependency_cross_check",
            *sink_config("cross-check", dependency_cross_check="atlahs-goal"),
        ),
        _refusal(
            "num_goal_ranks below NIC count",
            *sink_config("goal-ranks", num_goal_ranks=7),
        ),
        _refusal(
            "duplicate projection key",
            lambda: FabricGoalProjection(1, (row, row)),
        ),
    ]
    return {"refusals": refusals}


_DIGEST_BUILDERS = (
    ("worked_example_tp4_pp2_dp2", "placement", {"tp": 4, "pp": 2, "dp": 2}),
    ("m4_tp8", "placement", {"tp": 8}),
    ("rail_pp8", "pipeline", {}),
    ("m5_tp1_dp8", "placement", {"tp": 1, "dp": 8}),
    ("width_tp64", "placement", {"tp": 64, "nodes": 8, "gpus_per_node": 8}),
    ("fabric_one_plus_one", "fabric", {"render_physical_topology": True}),
    ("fabric_one_plus_one_disabled", "fabric", {"render_physical_topology": False}),
)


def _run_u5(output_root: Path, m5: Any) -> dict[str, Any]:
    from simllm.placement import (
        FabricTopologyManifest,
        declared_manifest,
        declared_pipeline_placement,
        declared_shared_nic_fabric,
        disaggregated_manifests,
    )

    root = output_root / "u5"
    root.mkdir(parents=True)
    digests = {}
    for label, kind, arguments in _DIGEST_BUILDERS:
        if kind == "placement":
            record = declared_manifest(**arguments)
        elif kind == "pipeline":
            record = declared_pipeline_placement(8)
        else:
            record = disaggregated_manifests(
                prefill_nodes=1, decode_nodes=1, **arguments
            ).fabric
        payload = record.save(root / f"{label}.json").read_bytes()
        digests[label] = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}

    placement = declared_manifest(tp=1, dp=U3_WORLD)
    round_trips = {}
    for gpus_per_nic in (1, 2, 4):
        fabric = declared_shared_nic_fabric(placement, gpus_per_nic=gpus_per_nic)
        path = fabric.save(root / f"shared-nic-g{gpus_per_nic}.json")
        loaded = FabricTopologyManifest.load(path)
        again = loaded.save(root / f"shared-nic-g{gpus_per_nic}-again.json")
        round_trips[f"unique-nic-g{gpus_per_nic}"] = {
            "byte_identical": again.read_bytes() == path.read_bytes(),
            "equal": loaded == fabric,
            "goal_rank_mapping": loaded.goal_rank_mapping,
        }
    gpu_spelling = replace(
        declared_shared_nic_fabric(placement, gpus_per_nic=1),
        goal_rank_mapping="gpu-rank",
    )
    path = gpu_spelling.save(root / "shared-nic-g1-gpu-rank.json")
    loaded = FabricTopologyManifest.load(path)
    loaded.validate()
    round_trips["gpu-rank-g1"] = {
        "byte_identical": loaded.save(root / "shared-nic-g1-gpu-rank-again.json").read_bytes()
        == path.read_bytes(),
        "equal": loaded == gpu_spelling,
        "goal_rank_mapping": loaded.goal_rank_mapping,
    }
    other_spelling = _refusal(
        "other goal_rank_mapping spelling",
        lambda: replace(gpu_spelling, goal_rank_mapping="nic-rank").validate(),
    )

    makespans: dict[str, dict[str, int]] = {}
    for shape, build in m5.STEP_SHAPES.items():
        for world in (2, 4, 8):
            _sink, result = _run_sink(
                output_root / "m5" / f"{shape}-ep{world}",
                dims=m5.moe_dims(world),
                ep_ranks=range(world),
                record=build(),
            )
            makespans.setdefault(shape, {})[str(world)] = result.step_latency_ps
    return {
        "digests": digests,
        "m5_makespans_ps": makespans,
        "other_spelling_refused": other_spelling["refused"],
        "round_trips": round_trips,
    }


def analyze_observation(
    observation: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen relations; fatal guards void, scored families count."""

    findings: list[str] = []
    fatal: dict[str, list[str]] = {
        "compatibility_digests": [],
        "m5_makespans": [],
        "projection_joins": [],
        "u3_invariants": [],
        "u5_round_trip": [],
    }
    baseline = frozen["baseline"]

    u5 = observation["u5"]
    for label, expected in baseline["artifacts"].items():
        if u5["digests"].get(label) != expected:
            fatal["compatibility_digests"].append(label)
    for shape, by_world in baseline["m5_makespans_ps"].items():
        for world, expected in by_world.items():
            if u5["m5_makespans_ps"].get(shape, {}).get(world) != expected:
                fatal["m5_makespans"].append(f"{shape} W={world}")
    for label, trip in u5["round_trips"].items():
        spelling = "gpu-rank" if label.startswith("gpu-rank") else "unique-nic"
        if not (trip["byte_identical"] and trip["equal"] and trip["goal_rank_mapping"] == spelling):
            fatal["u5_round_trip"].append(label)
    if not u5["other_spelling_refused"]:
        fatal["u5_round_trip"].append("other spelling accepted")

    identity_families: dict[str, bool] = {}
    u1 = observation["u1"]
    u1_holds = u1["goal_rank_identity"] and u1["nic_count"] == u1["rank_count"] == 16
    for shape, cell in u1["shapes"].items():
        if cell["joined_rows"] != cell["num_flows"] or cell["num_flows"] == 0:
            fatal["projection_joins"].append(f"u1 {shape}")
        holds = all(
            cell[name]
            for name in (
                "completion_files_identical",
                "goal_text_identical",
                "gpu_rank_keys_unique",
                "joined_endpoints_identity",
                "joined_times_identical",
                "outcomes_identical",
                "step_result_identical",
            )
        ) and cell["tag_multiplier"] == 1 and cell["gpu_rank_backend_rows"] == cell["num_flows"]
        if not holds:
            findings.append(f"u1 {shape}: identity does not hold")
        u1_holds = u1_holds and holds
    identity_families["u1_identity"] = u1_holds

    u2 = observation["u2"]
    u2_holds = u2["goal_rank_formula_holds"]
    for shape, cell in u2["shapes"].items():
        if cell["joined_rows"] != cell["num_flows"] or cell["num_flows"] == 0:
            fatal["projection_joins"].append(f"u2 {shape}")
        holds = all(
            cell[name]
            for name in (
                "goal_names_identical",
                "goal_text_differs",
                "goal_text_differs_only_in_rank_numbers",
                "joined_endpoints_permuted",
                "gpu_rank_keys_unique",
                "joined_times_identical",
                "outcomes_identical",
                "step_result_identical",
            )
        ) and cell["gpu_rank_backend_rows"] == cell["num_flows"]
        if not holds:
            findings.append(f"u2 {shape}: permutation identity does not hold")
        u2_holds = u2_holds and holds
    identity_families["u2_permutation"] = u2_holds

    cell = frozen["cells"]["u3_shared"]
    ps_per_byte = cell["ps_per_byte"]
    propagation_ps = cell["propagation_ps"]
    remote_pairs = 2 * (U3_WORLD // 2) ** 2
    local_pairs = U3_WORLD * (U3_WORLD - 1) - remote_pairs
    cases = observation["u3"]["cases"]
    relations = []
    for payload_bytes in cell["payload_bytes"]:
        reference = cases[f"s{payload_bytes}-gpu-rank-g1"]
        base = cases[f"s{payload_bytes}-unique-nic-g1"]
        for case in (reference, base):
            if case["joined_rows"] != case["backend_rows"] or case["backend_rows"] != remote_pairs:
                fatal["projection_joins"].append(f"u3 s{payload_bytes} {case['mode']}")
        if (
            base["goal_sha256"] != reference["goal_sha256"]
            or base["completion_rows_sha256"] != reference["completion_rows_sha256"]
        ):
            fatal["u3_invariants"].append(f"s{payload_bytes}: g=1 differs from gpu-rank")
        for gpus_per_nic in cell["gpus_per_nic"]:
            key = str(gpus_per_nic)
            case = cases[f"s{payload_bytes}-unique-nic-g{gpus_per_nic}"]
            label = f"s{payload_bytes} g={gpus_per_nic}"
            if case["joined_rows"] != case["backend_rows"] or case["backend_rows"] != remote_pairs:
                fatal["projection_joins"].append(f"u3 {label}")
            if not case["keys_unique_per_endpoint_pair"] or not case["quiescent"]:
                fatal["u3_invariants"].append(f"{label}: keys or quiescence")
            if case["fabric_bytes"] != remote_pairs * payload_bytes:
                fatal["u3_invariants"].append(f"{label}: fabric bytes moved")
            if case["nvlink_bytes"] != local_pairs * payload_bytes:
                fatal["u3_invariants"].append(f"{label}: NVLink bytes moved")
            if case["flow_start_ps"] != [0]:
                fatal["u3_invariants"].append(f"{label}: flows did not start together")
            floor_ps = payload_bytes * ps_per_byte + propagation_ps
            if any(fct < floor_ps for fct in case["flow_fct_ps"]):
                fatal["u3_invariants"].append(f"{label}: FCT below the serialization floor")
            flows = cell["flows_per_endpoint"][key]
            expected_ps = flows * payload_bytes * ps_per_byte + propagation_ps
            shape_holds = (
                case["goal_ranks"] == cell["goal_ranks"][key]
                and case["flows_per_endpoint_egress"] == [flows]
                and case["flows_per_endpoint_ingress"] == [flows]
                and case["tag_multiplier"] == gpus_per_nic**2
            )
            ratio = cell["ratio_to_g1"][key]
            fct_holds = (
                shape_holds
                and case["flow_fct_ps"] == [expected_ps]
                and len(base["flow_fct_ps"]) == 1
                and case["flow_fct_ps"][0] - propagation_ps
                == ratio * (base["flow_fct_ps"][0] - propagation_ps)
            )
            makespan_holds = (
                shape_holds
                and case["makespan_ps"] == expected_ps
                and case["makespan_ps"] - propagation_ps
                == ratio * (base["makespan_ps"] - propagation_ps)
            )
            for quantity, holds in (("fct", fct_holds), ("makespan", makespan_holds)):
                relations.append(
                    {
                        "expected_ps": expected_ps,
                        "gpus_per_nic": gpus_per_nic,
                        "holds": holds,
                        "payload_bytes": payload_bytes,
                        "quantity": quantity,
                    }
                )
                if not holds:
                    findings.append(f"u3 {label} {quantity}: frozen relation does not hold")

    refusals = observation["u4"]["refusals"]
    expected_labels = frozen["cells"]["u4_refusals"]
    if [item["label"] for item in refusals] != expected_labels:
        findings.append("u4: refusal inventory differs from the freeze")
    for item in refusals:
        if not (item["refused"] and item["workdir_absent"]):
            findings.append(f"u4 {item['label']}: not refused before any workdir")

    for guard, violations in fatal.items():
        findings.extend(f"fatal {guard}: {violation}" for violation in violations)
    violated_fatal = any(fatal.values())
    scored_ok = (
        all(item["holds"] for item in relations)
        and all(identity_families.values())
        and all(item["refused"] and item["workdir_absent"] for item in refusals)
        and [item["label"] for item in refusals] == expected_labels
    )
    status = "VOID" if violated_fatal else ("PASS" if scored_ok else "FAIL")
    return {
        "evidence": {
            "fatal_guards": {
                "compatibility_digests_checked": len(baseline["artifacts"]),
                "m5_makespans_checked": sum(len(item) for item in baseline["m5_makespans_ps"].values()),
                "violations": fatal,
            },
            "rejection_controls": {
                "refused": sum(item["refused"] and item["workdir_absent"] for item in refusals),
                "total": len(expected_labels),
            },
            "scored_identity_families": {
                "families": identity_families,
                "held": sum(identity_families.values()),
                "total": len(identity_families),
            },
            "scored_relation_instances": {
                "held": sum(item["holds"] for item in relations),
                "instances": relations,
                "total": len(relations),
            },
            "structural_guards": ["projection_joins", "u3_invariants", "u5_round_trip"],
        },
        "findings": findings,
        "status": status,
    }


def run_study(output_root: Path) -> dict[str, Any]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment["schema"] != "simllm-unique-nic-mapping-amendment-v1":
        raise SystemExit("unsupported expectations amendment")
    m5 = _m5_study()
    observation = {
        "u1": _run_u1(output_root, m5),
        "u2": _run_u2(output_root, m5),
        "u3": _run_u3(output_root, frozen),
        "u4": _run_u4(output_root, m5),
        "u5": _run_u5(output_root, m5),
    }
    analysis = analyze_observation(observation, frozen)
    return {
        "amendment_commit": AMENDMENT_COMMIT,
        "evidence": analysis["evidence"],
        "expectations_commit": EXPECTATIONS_COMMIT,
        "findings": analysis["findings"],
        "harness_sha256": _sha256_bytes(Path(__file__).resolve().read_bytes()),
        "implementation_commit": _git("rev-parse", "HEAD"),
        "observation": observation,
        "schema": RESULT_SCHEMA,
        "status": analysis["status"],
    }


def _fresh_output_root(requested: Path | None, label: str) -> Path:
    if requested is None:
        from simllm._local_config import path_from_env

        data_root = path_from_env(DATA_ROOT_ENV)
        if data_root is None:
            raise SystemExit(f"provide --output-root or set {DATA_ROOT_ENV}")
        requested = data_root / "unique_nic_mapping_v1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = requested.resolve() / f"{stamp}-{label}"
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _check(output_root: Path) -> None:
    if not TRACKED_RESULTS.is_file():
        raise SystemExit("tracked results.json is required")
    tracked_bytes = TRACKED_RESULTS.read_bytes()
    if b"\r" in tracked_bytes:
        raise SystemExit("tracked results.json must use LF line endings")
    tracked = json.loads(tracked_bytes)
    if tracked.get("schema") != RESULT_SCHEMA:
        raise SystemExit("results.json has an unsupported schema")
    if (tracked.get("expectations_commit"), tracked.get("amendment_commit")) != (
        EXPECTATIONS_COMMIT,
        AMENDMENT_COMMIT,
    ):
        raise SystemExit("results.json cites other expectations commits")
    if tracked.get("harness_sha256") != _sha256_bytes(Path(__file__).resolve().read_bytes()):
        raise SystemExit("the harness changed after results.json was produced")
    if not _is_ancestor(tracked["implementation_commit"]):
        raise SystemExit("the recorded implementation commit is not an ancestor of HEAD")
    fresh = run_study(output_root)
    (output_root / "results.json").write_bytes(_json_bytes(fresh))
    drift = [key for key in REPRODUCED_KEYS if fresh[key] != tracked[key]]
    if drift:
        raise SystemExit("results.json does not reproduce: " + ", ".join(drift))
    print(f"results.json reproduces; status={tracked['status']} sha256={_sha256_bytes(tracked_bytes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"external artifact directory; defaults under ${DATA_ROOT_ENV}",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _require_provenance()
    output_root = _fresh_output_root(args.output_root, "check" if args.check else "run")
    if args.check:
        _check(output_root)
        return
    result = run_study(output_root)
    payload = _json_bytes(result)
    (output_root / "results.json").write_bytes(payload)
    TRACKED_RESULTS.write_bytes(payload)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "findings")}
            | {
                "relations_held": result["evidence"]["scored_relation_instances"]["held"],
                "identity_families": result["evidence"]["scored_identity_families"]["families"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
