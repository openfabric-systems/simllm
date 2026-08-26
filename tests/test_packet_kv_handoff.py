from __future__ import annotations

import json
from pathlib import Path

from simllm.backends.htsim_rnic import (
    FlowCompletion,
    RnicRunResult,
)
from simllm.core import (
    PACKET_KV_HANDOFF_AUTHORITY,
    DeclaredKvHandoffPolicy,
    VirtualClock,
)
from simllm.traffic import PacketKvHandoffPolicy
from simllm.traffic import kv_handoff as kv_handoff_module


def _policy(tmp_path: Path) -> PacketKvHandoffPolicy:
    txt2bin = tmp_path / "txt2bin"
    htsim = tmp_path / "htsim_rnic"
    txt2bin.touch()
    htsim.touch()
    return PacketKvHandoffPolicy(
        artifact_dir=tmp_path / "packet",
        linkspeed_bps=400_000_000_000,
        txt2bin=txt2bin,
        htsim_rnic=htsim,
    )


def _install_backend_standins(monkeypatch) -> None:
    def convert(goal_path, binary_path, *, tool):
        del goal_path, tool
        Path(binary_path).write_bytes(b"goal-binary")
        return Path(binary_path)

    def run(config, *, binary, timeout_s):
        del binary, timeout_s
        flows = [
            FlowCompletion(
                profile="rnic-nn",
                flow_id=index,
                source=index,
                destination=index + 8,
                tag=6_200,
                payload_bytes=49_152,
                start_time_ps=20_000_000,
                completion_time_ps=21_000_000 + index,
                fct_ps=1_000_000 + index,
            )
            for index in range(8)
        ]
        return RnicRunResult(
            flows=flows,
            manifest=["[RNIC manifest] physical_quiescence=verified"],
            quiescent=True,
            goal_completion_time_ps=21_001_000,
        )

    monkeypatch.setattr(kv_handoff_module, "to_binary", convert)
    from simllm.backends import htsim_rnic as htsim_rnic_module

    monkeypatch.setattr(htsim_rnic_module, "run_htsim_rnic", run)


def test_packet_policy_renders_exact_pairs_and_prices_last_arrival(
    tmp_path,
    monkeypatch,
):
    _install_backend_standins(monkeypatch)
    policy = _policy(tmp_path)
    clock = VirtualClock(start_ps=7)

    event = policy.apply(
        clock,
        request_id="request-packet",
        kv_bytes=393_216,
    )

    assert event.authority == PACKET_KV_HANDOFF_AUTHORITY
    assert event.pricing_arm == "packet"
    assert event.submitted_at_ps == 7
    assert event.eligible_at_ps == event.started_at_ps == 20_000_007
    assert event.finished_at_ps == event.completed_at_ps == 21_000_014
    assert event.submission_ps == 20_000_000
    assert event.service_ps == 1_000_007
    assert event.total_ps == 21_000_007
    assert clock.now_ps == event.completed_at_ps

    artifact = policy.artifacts[0]
    assert artifact.chunk_bytes == (49_152,) * 8
    assert sum(artifact.chunk_bytes) == 393_216
    assert [
        (message.source_rank, message.destination_rank)
        for message in artifact.messages
    ] == [(index, index + 8) for index in range(8)]
    assert artifact.last_required_arrival_ps == max(
        flow.completion_time_ps for flow in artifact.flows
    )
    goal = artifact.goal_path.read_text()
    assert goal.count("calc 20000") == 8
    assert goal.count("send 49152b") == 8
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["aggregate_kv_bytes"] == 393_216
    assert manifest["packet_service_ps"] == 1_000_007


def test_constant_and_off_arms_create_no_packet_artifacts(tmp_path):
    clock = VirtualClock()
    DeclaredKvHandoffPolicy(100).apply(
        clock,
        request_id="constant",
        kv_bytes=8,
    )
    DeclaredKvHandoffPolicy.off().apply(
        clock,
        request_id="off",
        kv_bytes=8,
    )

    assert list(tmp_path.iterdir()) == []


def test_packet_policy_partitions_nondivisible_bytes_exactly(tmp_path):
    policy = _policy(tmp_path)

    assert policy._chunks(11) == (2, 2, 2, 1, 1, 1, 1, 1)
