"""PLACE-2: unique-nic GOAL-rank mapping, collapsed-pair tags and the projection join."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.backends.htsim_rnic import FlowCompletion, RnicRunResult
from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import (
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    NicFabricPlacement,
    RankMapper,
    declared_manifest,
    declared_pipeline_placement,
    declared_shared_nic_fabric,
    disaggregated_manifests,
)
from simllm.traffic import (
    CollectiveCommunicationPhase,
    DirectedCollectiveSegment,
    FabricGoalProjection,
    FabricSegmentProjection,
    classify_step_locality,
    plan_fabric_goal_projection,
    render_fabric_phase_goal,
    step_communication_phases,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "unique_nic_mapping_v1"
PROPAGATION_PS = 2_000_000
PS_PER_BYTE = 20


def _expectations() -> dict:
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def _moe_dims(world: int, *, layers: int = 2) -> ModelDims:
    """The m5 granite geometry at EP width ``world`` with a short layer count."""

    return ModelDims(
        num_layers=layers,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32 // world,
    )


def _decode_record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                f"d{index}",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=2048,
            )
            for index in range(8)
        ],
    )


def _one_plus_one():
    manifests = disaggregated_manifests(
        prefill_nodes=1,
        decode_nodes=1,
        render_physical_topology=False,
    )
    return manifests.placement, replace(manifests.fabric, goal_rank_mapping="unique-nic")


def _reversed_fabric(placement) -> FabricTopologyManifest:
    """One NIC per GPU, NIC ``j`` affine to local rank ``width - 1 - j``."""

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
        gpus = tuple(
            GpuFabricPlacement(
                global_rank=rank.global_rank,
                gpu_id=f"sim-gpu-{rank.global_rank:04d}",
                node_id=host,
                pcie_location=f"{host}/pcie-{rank.local_rank}",
                nic_id=nic_ids[width - 1 - rank.local_rank],
            )
            for rank in ranks
        )
        nics = tuple(
            NicFabricPlacement(
                nic_id=nic_ids[index],
                node_id=host,
                fabric_location=f"{host}/nic-{index}",
                affine_gpu_rank=ranks[width - 1 - index].global_rank,
            )
            for index in range(width)
        )
        nodes.append(FabricNodePlacement(host, "declared", gpus, nics))
    return FabricTopologyManifest(nodes=nodes, goal_rank_mapping="unique-nic")


def _all_pairs_phase(payload_bytes: int, *, world: int = 16, tag: int = 1000):
    ranks = tuple(range(world))
    return CollectiveCommunicationPhase(
        phase_id="u3:all-pairs",
        layer=0,
        participants=ranks,
        segments=tuple(
            DirectedCollectiveSegment(source, destination, payload_bytes, tag)
            for source in ranks
            for destination in ranks
            if source != destination
        ),
        operation_id="u3:all-pairs",
    )


def _canonical_goal(text: str, goal_to_semantic: dict[int, int]) -> dict[int, list[str]]:
    """Rank blocks keyed by semantic rank with peer ranks mapped back."""

    blocks: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines()[1:]:
        if line.startswith("rank "):
            current = goal_to_semantic[int(line.split()[1])]
            blocks[current] = []
        elif line != "}":
            words = line.partition(": ")[2].split()
            if words[0] in ("send", "recv"):
                words[3] = str(goal_to_semantic[int(words[3])])
            blocks[current].append(" ".join(words))
    return blocks


def _fake_fluid_backend(monkeypatch) -> None:
    """Typed completion rows from an endpoint-symmetric closed form."""

    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def run(config):
        flows = []
        rank = None
        for line in Path(config.goal_bin).read_text().splitlines():
            if line.startswith("rank "):
                rank = int(line.split()[1])
            elif ": send " in line:
                words = line.partition(": ")[2].split()
                size = int(words[1].removesuffix("b"))
                completion = size * PS_PER_BYTE + PROPAGATION_PS
                flows.append(
                    FlowCompletion(
                        "rnic-nn-fluid",
                        len(flows),
                        rank,
                        int(words[3]),
                        int(words[5]),
                        size,
                        0,
                        completion,
                        completion,
                    )
                )
        return RnicRunResult(flows, [], True)

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", run)


def _sink(tmp_path, label, placement, *, world, fabric=None) -> HtsimStepSink:
    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_moe_dims(world),
            workdir=tmp_path / label,
            ep_ranks=tuple(range(world)),
            placement_manifest=placement,
            goal_rank_mapping="gpu-rank" if fabric is None else "unique-nic",
            fabric_manifest=fabric,
        )
    )


def test_freeze_pins_schema_interface_and_u3_table():
    frozen = _expectations()
    assert frozen["schema"] == "simllm-unique-nic-mapping-expectations-v1"
    assert frozen["task"] == "PLACE-2"
    assert frozen["baseline"]["commit"] == "26704c1c18e3c5503cb717cdd31391b82d3983a0"
    assert frozen["baseline"]["m5_makespans_ps"] == {
        "decode8x2048": {"2": 563_362_560, "4": 486_963_888, "8": 448_764_528},
        "prefill2048": {
            "2": 17_592_951_360,
            "4": 25_646_015_088,
            "8": 29_672_546_928,
        },
    }
    interface = frozen["interface"]
    assert interface["mapper_modes"] == ["gpu-rank", "unique-nic"]
    assert interface["mapper_new_methods"] == ["nic_of"]
    assert interface["shared_nic_builder"] == "declared_shared_nic_fabric"
    assert interface["sink_fields"] == ["goal_rank_mapping", "fabric_manifest"]

    u3 = frozen["cells"]["u3_shared"]
    assert u3["gpus_per_nic"] == [1, 2, 4]
    assert u3["payload_bytes"] == [65_536, 1_048_576]
    assert u3["ps_per_byte"] == 8 * 10**12 // u3["linkspeed_bps"] == PS_PER_BYTE
    assert u3["propagation_ps"] == PROPAGATION_PS
    for gpus_per_nic in u3["gpus_per_nic"]:
        key = str(gpus_per_nic)
        assert u3["flows_per_endpoint"][key] == 8 * gpus_per_nic
        assert u3["goal_ranks"][key] == 16 // gpus_per_nic
        assert u3["ratio_to_g1"][key] == gpus_per_nic
    assert frozen["evidence"]["scored_relation_instances"] == (
        len(u3["payload_bytes"]) * len(u3["gpus_per_nic"]) * 2
    )
    assert frozen["evidence"]["fatal_compatibility_digests"] == 7
    assert frozen["evidence"]["fatal_makespan_literals"] == 6
    assert len(frozen["cells"]["u4_refusals"]) == 10

    amendment = json.loads(
        (STUDY / "expectations-amendment-2026-09-13.json").read_text(encoding="utf-8")
    )
    assert amendment["schema"] == "simllm-unique-nic-mapping-amendment-v1"
    assert amendment["u3_path"] == [
        "classify_step_locality",
        "render_fabric_phase_goal",
        "to_binary",
        "run_htsim_rnic",
        "projection join",
    ]
    assert amendment["m_rule"] == "per phase, maximized over the step"
    assert amendment["u2_workload"] == ["decode8x2048", "prefill2048"]


_DIGEST_BUILDERS = {
    "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
    "m4_tp8": lambda: declared_manifest(tp=8),
    "rail_pp8": lambda: declared_pipeline_placement(8),
    "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
    "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
    "fabric_one_plus_one": lambda: disaggregated_manifests(
        prefill_nodes=1, decode_nodes=1
    ).fabric,
    "fabric_one_plus_one_disabled": lambda: disaggregated_manifests(
        prefill_nodes=1, decode_nodes=1, render_physical_topology=False
    ).fabric,
}


@pytest.mark.parametrize("label", sorted(_DIGEST_BUILDERS))
def test_default_mapping_keeps_the_seven_compatibility_digests(tmp_path, label):
    record = _expectations()["baseline"]["artifacts"][label]
    payload = _DIGEST_BUILDERS[label]().save(tmp_path / f"{label}.json").read_bytes()
    assert len(payload) == record["bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_shared_nic_builder_orders_nics_and_affinity():
    placement = declared_manifest(tp=1, dp=16)
    fabric = declared_shared_nic_fabric(placement, gpus_per_nic=4)
    mapper = RankMapper(placement, mode="unique-nic", fabric=fabric)

    assert not fabric.physical_rendering_enabled
    assert [nic.nic_id for node in fabric.nodes for nic in node.nics] == [
        "node-0:nic-0",
        "node-0:nic-1",
        "node-1:nic-0",
        "node-1:nic-1",
    ]
    assert mapper.num_goal_ranks() == 4
    for rank in range(16):
        assert mapper.nic_of(rank) == f"node-{rank // 8}:nic-{(rank % 8) // 4}"
        assert mapper.goal_rank(rank) == 2 * (rank // 8) + (rank % 8) // 4
    assert RankMapper(placement).goal_rank(9) == 9
    with pytest.raises(ValueError, match="requires a fabric"):
        RankMapper(placement).nic_of(0)


def test_u1_identity_mapper_and_renderer_are_byte_identical():
    placement, fabric = _one_plus_one()
    gpu = RankMapper(placement)
    nic = RankMapper(placement, mode="unique-nic", fabric=fabric)
    assert nic.num_goal_ranks() == gpu.num_goal_ranks() == 16
    for rank in range(16):
        assert nic.goal_rank(rank) == rank
        assert nic.nic_of(rank) == f"sim-nic-{rank:04d}"

    phases = step_communication_phases(
        _decode_record(), _moe_dims(16), (0,), ep_ranks=tuple(range(16))
    )
    plan = classify_step_locality(phases, rank_mapper=nic)
    assert plan == classify_step_locality(phases, rank_mapper=gpu)
    gpu_projection = plan_fabric_goal_projection(plan.phases, rank_mapper=gpu)
    nic_projection = plan_fabric_goal_projection(plan.phases, rank_mapper=nic)
    assert gpu_projection == nic_projection
    assert nic_projection.tag_multiplier == 1
    rendered = 0
    for phase in plan.phases:
        if not phase.fabric_segments:
            continue
        standalone = render_fabric_phase_goal(phase, rank_mapper=gpu).render()
        assert render_fabric_phase_goal(
            phase, rank_mapper=gpu, projection=gpu_projection
        ).render() == standalone
        assert render_fabric_phase_goal(
            phase, rank_mapper=nic, projection=nic_projection
        ).render() == standalone
        rendered += 1
    assert rendered == 4


def test_u2_permutation_moves_only_goal_rank_numbers():
    placement = declared_manifest(tp=1, dp=8, gpus_per_node=4)
    fabric = _reversed_fabric(placement)
    gpu = RankMapper(placement)
    nic = RankMapper(placement, mode="unique-nic", fabric=fabric)
    for rank in placement.ranks:
        node = int(rank.hostname.removeprefix("node-"))
        assert nic.goal_rank(rank.global_rank) == 4 * node + (3 - rank.local_rank)
    inverse = {nic.goal_rank(rank): rank for rank in range(8)}

    phases = step_communication_phases(
        _decode_record(), _moe_dims(8), (0,), ep_ranks=tuple(range(8))
    )
    plan = classify_step_locality(phases, rank_mapper=nic)
    assert plan == classify_step_locality(phases, rank_mapper=gpu)
    projection = plan_fabric_goal_projection(plan.phases, rank_mapper=nic)
    assert projection.tag_multiplier == 1
    differing = 0
    for phase in plan.phases:
        if not phase.fabric_segments:
            continue
        baseline = render_fabric_phase_goal(phase, rank_mapper=gpu).render()
        permuted = render_fabric_phase_goal(
            phase, rank_mapper=nic, projection=projection
        ).render()
        differing += baseline != permuted
        assert _canonical_goal(permuted, inverse) == _canonical_goal(
            baseline, {rank: rank for rank in range(8)}
        )
    assert differing == 4


@pytest.mark.parametrize("gpus_per_nic", [1, 2, 4])
def test_u3_shared_endpoints_follow_the_frozen_counts(gpus_per_nic):
    payload_bytes = 65_536
    placement = declared_manifest(tp=1, dp=16)
    fabric = declared_shared_nic_fabric(placement, gpus_per_nic=gpus_per_nic)
    gpu = RankMapper(placement)
    nic = RankMapper(placement, mode="unique-nic", fabric=fabric)

    plan = classify_step_locality((_all_pairs_phase(payload_bytes),), rank_mapper=nic)
    assert plan == classify_step_locality(
        (_all_pairs_phase(payload_bytes),), rank_mapper=gpu
    )
    assert plan.fabric_bytes == 128 * payload_bytes
    assert plan.nvlink_bytes == 112 * payload_bytes
    projection = plan_fabric_goal_projection(plan.phases, rank_mapper=nic)
    assert projection.tag_multiplier == gpus_per_nic**2
    trace = render_fabric_phase_goal(plan.phases[0], rank_mapper=nic, projection=projection)

    assert trace.num_ranks == nic.num_goal_ranks() == 16 // gpus_per_nic
    egress = Counter(message.source_rank for message in trace.messages)
    ingress = Counter(message.destination_rank for message in trace.messages)
    assert set(egress.values()) == set(ingress.values()) == {8 * gpus_per_nic}
    keys = {
        (message.source_rank, message.destination_rank, message.tag)
        for message in trace.messages
    }
    assert len(keys) == len(trace.messages) == 128
    assert {row.rendered_tag - 1000 * gpus_per_nic**2 for row in projection.rows} == set(
        range(gpus_per_nic**2)
    )
    if gpus_per_nic == 1:
        assert trace.render() == render_fabric_phase_goal(
            plan.phases[0], rank_mapper=gpu
        ).render()


def test_u4_mapper_and_builder_refusals():
    placement = declared_manifest(tp=1, dp=16)
    fabric = declared_shared_nic_fabric(placement, gpus_per_nic=2)

    with pytest.raises(ValueError, match="requires a fabric"):
        RankMapper(placement, mode="unique-nic")
    with pytest.raises(ValueError, match="goal_rank_mapping is unique-nic"):
        RankMapper(
            placement,
            mode="unique-nic",
            fabric=replace(fabric, goal_rank_mapping="gpu-rank"),
        )
    node = fabric.nodes[0]
    broken = replace(node.gpus[0], nic_id="node-0:nic-missing")
    missing_nic = replace(
        fabric,
        nodes=[replace(node, gpus=(broken, *node.gpus[1:])), *fabric.nodes[1:]],
    )
    with pytest.raises(ValueError, match="names no NIC"):
        RankMapper(placement, mode="unique-nic", fabric=missing_nic)
    other = declared_shared_nic_fabric(declared_manifest(tp=1, dp=8), gpus_per_nic=2)
    with pytest.raises(ValueError, match="GPU set"):
        RankMapper(placement, mode="unique-nic", fabric=other)
    with pytest.raises(ValueError, match="does not divide"):
        declared_shared_nic_fabric(placement, gpus_per_nic=3)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"peer_packet": object()}, "peer_packet"),
        ({"topology": Path("clos.topo")}, "topology"),
        ({"flow_session": object()}, "flow_session"),
        ({"dependency_cross_check": "atlahs-goal"}, "dependency_cross_check"),
        ({"num_goal_ranks": 7}, "NIC count"),
        ({"fabric_manifest": None}, "requires fabric_manifest"),
        ({"profile": "rnic-cn"}, "null-network"),
    ],
)
def test_u4_sink_refuses_unmodeled_seams_before_any_workdir(tmp_path, override, match):
    placement = declared_manifest(tp=1, dp=16)
    workdir = tmp_path / "never-created"
    values = {
        "profile": "rnic-nn-fluid",
        "tp_ranks": (0,),
        "dims": _moe_dims(16),
        "workdir": workdir,
        "ep_ranks": tuple(range(16)),
        "placement_manifest": placement,
        "goal_rank_mapping": "unique-nic",
        "fabric_manifest": declared_shared_nic_fabric(placement, gpus_per_nic=2),
    }
    values.update(override)
    with pytest.raises(ValueError, match=match):
        HtsimStepSinkConfig(**values)
    assert not workdir.exists()


def test_u4_projection_refuses_duplicate_keys_and_unjoined_rows():
    segment = DirectedCollectiveSegment(0, 8, 64, 1000)
    row = FabricSegmentProjection(0, "phase", 0, 4, 1000, segment)
    with pytest.raises(ValueError, match="duplicate key"):
        FabricGoalProjection(1, (row, row))
    with pytest.raises(ValueError, match="tag rule"):
        FabricGoalProjection(1, (replace(row, rendered_tag=1001),))

    projection = FabricGoalProjection(1, (row,))
    completion = SimpleNamespace(source=0, destination=4, tag=1000, payload_bytes=64)
    assert projection.join_completions([completion]) == ((completion, row),)
    with pytest.raises(ValueError, match="duplicate projection key"):
        projection.join_completions([completion, completion])
    with pytest.raises(ValueError, match="missing completion"):
        projection.join_completions([])
    with pytest.raises(ValueError, match="joins no fabric segment"):
        projection.join_completions(
            [SimpleNamespace(source=4, destination=0, tag=1000, payload_bytes=64)]
        )
    with pytest.raises(ValueError, match="payload"):
        projection.join_completions(
            [SimpleNamespace(source=0, destination=4, tag=1000, payload_bytes=65)]
        )


def test_u5_shared_nic_fabric_round_trips_both_spellings(tmp_path):
    placement = declared_manifest(tp=1, dp=16)
    for gpus_per_nic in (1, 2, 4):
        fabric = declared_shared_nic_fabric(placement, gpus_per_nic=gpus_per_nic)
        path = fabric.save(tmp_path / f"shared-{gpus_per_nic}.json")
        loaded = FabricTopologyManifest.load(path)
        assert loaded == fabric
        assert loaded.goal_rank_mapping == "unique-nic"
        again = loaded.save(tmp_path / f"shared-{gpus_per_nic}-again.json")
        assert again.read_bytes() == path.read_bytes()

    gpu_spelling = replace(
        declared_shared_nic_fabric(placement, gpus_per_nic=1),
        goal_rank_mapping="gpu-rank",
    )
    loaded = FabricTopologyManifest.load(gpu_spelling.save(tmp_path / "gpu-rank.json"))
    loaded.validate()
    assert loaded.goal_rank_mapping == "gpu-rank"
    with pytest.raises(ValueError, match="gpu-rank or unique-nic"):
        replace(gpu_spelling, goal_rank_mapping="nic-rank").validate()


def test_u1_sink_identity_through_typed_completion_rows(tmp_path, monkeypatch):
    _fake_fluid_backend(monkeypatch)
    placement, fabric = _one_plus_one()
    gpu = _sink(tmp_path, "gpu", placement, world=16)
    nic = _sink(tmp_path, "nic", placement, world=16, fabric=fabric)

    assert gpu(_decode_record()) == nic(_decode_record())
    assert gpu.outcomes == nic.outcomes
    assert gpu.locality_outcomes == nic.locality_outcomes
    names = sorted(path.name for path in (tmp_path / "gpu").glob("*.goal"))
    assert names
    assert names == sorted(path.name for path in (tmp_path / "nic").glob("*.goal"))
    for name in names:
        assert (tmp_path / "gpu" / name).read_bytes() == (tmp_path / "nic" / name).read_bytes()
    assert gpu.fabric_join_outcomes == []
    nic_join = nic.fabric_join_outcomes[0]
    assert nic_join.goal_rank_mapping == "unique-nic"
    assert nic_join.tag_multiplier == 1
    assert len(nic_join.segments) == nic.outcomes[0].num_flows
    for item in nic_join.segments:
        assert (item.goal_source_rank, item.goal_destination_rank) == (
            item.source_rank,
            item.destination_rank,
        )
        assert item.completion_time_ps == item.payload_bytes * PS_PER_BYTE + PROPAGATION_PS


def test_u2_sink_permutation_keeps_results_and_joined_times(tmp_path, monkeypatch):
    _fake_fluid_backend(monkeypatch)
    placement = declared_manifest(tp=1, dp=8, gpus_per_node=4)
    fabric = _reversed_fabric(placement)
    gpu = _sink(tmp_path, "gpu", placement, world=8)
    nic = _sink(tmp_path, "nic", placement, world=8, fabric=fabric)

    assert gpu(_decode_record()) == nic(_decode_record())
    assert gpu.outcomes == nic.outcomes

    assert gpu.fabric_join_outcomes == []
    segments = nic.fabric_join_outcomes[0].segments
    assert len(segments) == nic.outcomes[0].num_flows > 0
    mapper = RankMapper(placement, mode="unique-nic", fabric=fabric)
    for item in segments:
        assert item.completion_time_ps == item.payload_bytes * PS_PER_BYTE + PROPAGATION_PS
        assert (item.goal_source_rank, item.goal_destination_rank) == (
            mapper.goal_rank(item.source_rank),
            mapper.goal_rank(item.destination_rank),
        )


def test_shared_nic_sink_collapses_pairs_and_joins_every_row_once(tmp_path, monkeypatch):
    _fake_fluid_backend(monkeypatch)
    placement = declared_manifest(tp=1, dp=16)
    fabric = declared_shared_nic_fabric(placement, gpus_per_nic=2)
    sink = _sink(tmp_path, "shared", placement, world=16, fabric=fabric)
    sink(_decode_record())

    joined = sink.fabric_join_outcomes[0]
    assert joined.tag_multiplier == 2
    keys = [
        (item.goal_source_rank, item.goal_destination_rank, item.rendered_tag)
        for item in joined.segments
    ]
    assert len(keys) == len(set(keys)) == sink.outcomes[0].num_flows
    assert {item.rendered_tag - 2 * item.tag for item in joined.segments} == {0, 1}
    goal_files = sorted((tmp_path / "shared").glob("*.goal"))
    assert goal_files
    assert all(path.read_text().startswith("num_ranks 8\n") for path in goal_files)


def test_default_path_keeps_unjoined_rows_and_unique_nic_refuses_them(tmp_path, monkeypatch):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    unrelated = FlowCompletion("rnic-nn-fluid", 0, 0, 1, 7, 1, 0, 5, 5)
    monkeypatch.setattr(
        step_sink_module,
        "run_htsim_rnic",
        lambda config: RnicRunResult([unrelated], [], True),
    )
    placement, fabric = _one_plus_one()
    gpu = _sink(tmp_path, "gpu", placement, world=16)
    assert gpu(_decode_record()) is not None
    assert gpu.fabric_join_outcomes == []

    nic = _sink(tmp_path, "nic", placement, world=16, fabric=fabric)
    with pytest.raises(ValueError, match="joins no fabric segment"):
        nic(_decode_record())
