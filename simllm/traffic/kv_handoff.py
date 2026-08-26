"""Packet rendering and timing authority for a disaggregated KV handoff."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from simllm.core import PACKET_KV_HANDOFF_AUTHORITY, KvHandoffEvent, VirtualClock
from simllm.goal import GoalMessage, GoalTrace, to_binary
from simllm.traffic.patterns import pairwise_all_to_allv

if TYPE_CHECKING:
    from simllm.backends.htsim_rnic import FlowCompletion

PACKET_KV_HANDOFF_SCHEMA = "simllm-packet-kv-handoff-artifact-v1"
DEFAULT_PACKET_KV_TAG = 6_200


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _rank_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not value:
        raise ValueError(f"{name} must not be empty")
    for rank in value:
        if isinstance(rank, bool) or type(rank) is not int:
            raise TypeError(f"{name} must contain integers")
        if rank < 0:
            raise ValueError(f"{name} must contain nonnegative ranks")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must contain distinct ranks")
    return value


def _message_json(message: GoalMessage) -> dict[str, object]:
    return {
        "operation_id": message.operation_id,
        "source_rank": message.source_rank,
        "destination_rank": message.destination_rank,
        "payload_bytes": message.payload_bytes,
        "tag": message.tag,
        "send_label": message.send_label,
        "receive_label": message.receive_label,
        "request_payload_bytes": [list(row) for row in message.request_payload_bytes],
    }


@dataclass(frozen=True)
class PacketKvHandoffArtifact:
    """Read-only projection of one rendered and completed packet handoff."""

    request_id: str
    aggregate_kv_bytes: int
    chunk_bytes: tuple[int, ...]
    messages: tuple[GoalMessage, ...]
    flows: tuple[FlowCompletion, ...]
    goal_path: Path
    goal_binary_path: Path
    completion_csv_path: Path
    manifest_path: Path
    goal_completion_time_ps: int | None
    first_packet_start_ps: int
    last_required_arrival_ps: int
    quiescent: bool

    @property
    def packet_service_ps(self) -> int:
        return self.last_required_arrival_ps - self.first_packet_start_ps

    def to_json(self) -> dict[str, object]:
        return {
            "schema": PACKET_KV_HANDOFF_SCHEMA,
            "request_id": self.request_id,
            "aggregate_kv_bytes": self.aggregate_kv_bytes,
            "chunk_bytes": list(self.chunk_bytes),
            "messages": [_message_json(message) for message in self.messages],
            "flows": [asdict(flow) for flow in self.flows],
            "artifacts": {
                "goal": self.goal_path.name,
                "goal_binary": self.goal_binary_path.name,
                "completion_csv": self.completion_csv_path.name,
                "manifest": self.manifest_path.name,
            },
            "goal_completion_time_ps": self.goal_completion_time_ps,
            "first_packet_start_ps": self.first_packet_start_ps,
            "last_required_arrival_ps": self.last_required_arrival_ps,
            "packet_service_ps": self.packet_service_ps,
            "quiescent": self.quiescent,
        }


@dataclass
class PacketKvHandoffPolicy:
    """Render one request's rank-local KV shards and price their last arrival."""

    artifact_dir: Path
    linkspeed_bps: int
    txt2bin: Path
    htsim_rnic: Path
    pcie_submission_ps: int = 20_000_000
    prefill_ranks: tuple[int, ...] = tuple(range(8))
    decode_ranks: tuple[int, ...] = tuple(range(8, 16))
    profile: str = "rnic-nn"
    tag: int = DEFAULT_PACKET_KV_TAG
    timeout_s: int = 600
    artifacts: list[PacketKvHandoffArtifact] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_dir, Path):
            raise TypeError("artifact_dir must be a Path")
        for name in ("txt2bin", "htsim_rnic"):
            path = getattr(self, name)
            if not isinstance(path, Path):
                raise TypeError(f"{name} must be a Path")
            if not path.is_file():
                raise FileNotFoundError(f"{name} does not exist: {path}")
        _positive_int("linkspeed_bps", self.linkspeed_bps)
        _positive_int("pcie_submission_ps", self.pcie_submission_ps)
        if self.pcie_submission_ps % 1_000:
            raise ValueError("pcie_submission_ps must be a whole GOAL nanosecond")
        self.prefill_ranks = _rank_tuple("prefill_ranks", self.prefill_ranks)
        self.decode_ranks = _rank_tuple("decode_ranks", self.decode_ranks)
        if len(self.prefill_ranks) != len(self.decode_ranks):
            raise ValueError("prefill and decode rank sets must have equal width")
        if set(self.prefill_ranks) & set(self.decode_ranks):
            raise ValueError("prefill and decode rank sets must be disjoint")
        if self.profile != "rnic-nn":
            raise ValueError("the accepted packet KV arm requires profile='rnic-nn'")
        _positive_int("tag", self.tag)
        _positive_int("timeout_s", self.timeout_s)

    def _chunks(self, kv_bytes: int) -> tuple[int, ...]:
        kv_bytes = _positive_int("kv_bytes", kv_bytes)
        width = len(self.prefill_ranks)
        if kv_bytes < width:
            raise ValueError("kv_bytes must permit one positive chunk per rank pair")
        quotient, remainder = divmod(kv_bytes, width)
        chunks = tuple(
            quotient + (1 if index < remainder else 0) for index in range(width)
        )
        if any(chunk <= 0 for chunk in chunks) or sum(chunks) != kv_bytes:
            raise RuntimeError("KV chunk partition did not conserve aggregate bytes")
        return chunks

    def _request_dir(self, request_id: str) -> Path:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a nonblank string")
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
        return self.artifact_dir / f"request-{digest}"

    def _render(
        self,
        request_id: str,
        chunks: tuple[int, ...],
    ) -> tuple[GoalTrace, Path, Path, Path, Path]:
        request_dir = self._request_dir(request_id)
        request_dir.mkdir(parents=True, exist_ok=False)
        goal_path = request_dir / "kv-handoff.goal"
        binary_path = request_dir / "kv-handoff.bin"
        completion_path = request_dir / "flow-completions.csv"
        manifest_path = request_dir / "packet-handoff.json"

        trace = GoalTrace(max((*self.prefill_ranks, *self.decode_ranks)) + 1)
        operation_id = (
            f"kv-handoff-{hashlib.sha256(request_id.encode()).hexdigest()[:16]}"
        )
        after = {
            rank: trace.rank(rank).calc(
                self.pcie_submission_ps // 1_000,
                operation_id=operation_id,
            )
            for rank in self.prefill_ranks
        }
        send_bytes = {
            (source, destination): chunk
            for source, destination, chunk in zip(
                self.prefill_ranks,
                self.decode_ranks,
                chunks,
                strict=True,
            )
        }
        pairwise_all_to_allv(
            trace,
            [*self.prefill_ranks, *self.decode_ranks],
            send_bytes,
            self.tag,
            after,
            operation_id=operation_id,
            request_send_bytes={
                pair: ((request_id, payload_bytes),)
                for pair, payload_bytes in send_bytes.items()
            },
        )
        trace.write(goal_path)
        to_binary(goal_path, binary_path, tool=self.txt2bin)
        return trace, goal_path, binary_path, completion_path, manifest_path

    @staticmethod
    def _validate_flow_projection(
        messages: tuple[GoalMessage, ...],
        flows: tuple[FlowCompletion, ...],
    ) -> None:
        expected = sorted(
            (
                message.source_rank,
                message.destination_rank,
                message.tag,
                message.payload_bytes,
            )
            for message in messages
        )
        observed = sorted(
            (flow.source, flow.destination, flow.tag, flow.payload_bytes)
            for flow in flows
        )
        if observed != expected:
            raise RuntimeError(
                "backend flows do not conserve rendered endpoints, tags and bytes"
            )

    def schedule(
        self,
        *,
        submitted_at_ps: int,
        request_id: str,
        kv_bytes: int,
    ) -> KvHandoffEvent:
        """Run the packet arm and return its sole immutable timing projection."""

        if isinstance(submitted_at_ps, bool) or type(submitted_at_ps) is not int:
            raise TypeError("submitted_at_ps must be an integer")
        if submitted_at_ps < 0:
            raise ValueError("submitted_at_ps must be nonnegative")
        chunks = self._chunks(kv_bytes)
        from simllm.backends.htsim_rnic import HtsimRnicConfig, run_htsim_rnic

        trace, goal_path, binary_path, completion_path, manifest_path = self._render(
            request_id,
            chunks,
        )
        run = run_htsim_rnic(
            HtsimRnicConfig(
                goal_bin=binary_path,
                profile=self.profile,
                linkspeed_bps=self.linkspeed_bps,
                completion_csv=completion_path,
            ),
            binary=self.htsim_rnic,
            timeout_s=self.timeout_s,
        )
        flows = tuple(run.flows)
        messages = trace.messages
        if len(messages) != len(self.prefill_ranks) or len(flows) != len(messages):
            raise RuntimeError("packet handoff must produce one flow per rank pair")
        self._validate_flow_projection(messages, flows)
        first_packet_start = min(flow.start_time_ps for flow in flows)
        last_required_arrival = max(flow.completion_time_ps for flow in flows)
        if first_packet_start != self.pcie_submission_ps:
            raise RuntimeError(
                "first packet service did not start at the declared PCIe boundary"
            )
        if any(flow.start_time_ps < self.pcie_submission_ps for flow in flows):
            raise RuntimeError("a packet started before PCIe submission completed")
        if last_required_arrival <= first_packet_start:
            raise RuntimeError("packet handoff has no positive service interval")

        artifact = PacketKvHandoffArtifact(
            request_id=request_id,
            aggregate_kv_bytes=kv_bytes,
            chunk_bytes=chunks,
            messages=messages,
            flows=flows,
            goal_path=goal_path,
            goal_binary_path=binary_path,
            completion_csv_path=completion_path,
            manifest_path=manifest_path,
            goal_completion_time_ps=run.goal_completion_time_ps,
            first_packet_start_ps=first_packet_start,
            last_required_arrival_ps=last_required_arrival,
            quiescent=run.quiescent,
        )
        manifest_path.write_text(
            json.dumps(artifact.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.artifacts.append(artifact)
        return KvHandoffEvent(
            request_id=request_id,
            kv_bytes=kv_bytes,
            submitted_at_ps=submitted_at_ps,
            eligible_at_ps=submitted_at_ps + self.pcie_submission_ps,
            started_at_ps=submitted_at_ps + first_packet_start,
            finished_at_ps=submitted_at_ps + last_required_arrival,
            completed_at_ps=submitted_at_ps + last_required_arrival,
            pricing_arm="packet",
            authority=PACKET_KV_HANDOFF_AUTHORITY,
        )

    def apply(
        self,
        clock: VirtualClock,
        *,
        request_id: str,
        kv_bytes: int,
    ) -> KvHandoffEvent:
        """Run one packet handoff and advance the session to its last arrival."""

        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        event = self.schedule(
            submitted_at_ps=clock.now_ps,
            request_id=request_id,
            kv_bytes=kv_bytes,
        )
        clock.advance_to(event.completed_at_ps)
        return event


__all__ = [
    "DEFAULT_PACKET_KV_TAG",
    "PACKET_KV_HANDOFF_SCHEMA",
    "PacketKvHandoffArtifact",
    "PacketKvHandoffPolicy",
]
