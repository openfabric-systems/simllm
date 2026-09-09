"""Persistent native packet ownership for independent serving cache transfers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import astuple, replace
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import quote

from simllm.core.clock import VirtualClock
from simllm.core.pd_session import (
    KvHandoffJoin,
    KvHandoffShardCompletion,
    PendingKvHandoff,
    ServingPoolRole,
)
from simllm.placement.disaggregated import DisaggregatedDeploymentManifests

if TYPE_CHECKING:
    from simllm.backends.flow_session import FlowSessionConfig


class SharedKvHandoffRuntime:
    """One native session and read-only request joins over its accepted flows.

    Engine nodes are explicit bindings into the standard deployment manifest.
    The driver supplies actual local worker ranks when binding the public clock.
    Neither submission nor packet progress advances that clock.
    """

    def __init__(
        self, config: FlowSessionConfig, command: Sequence[str], *, session_id: str,
        deployment: DisaggregatedDeploymentManifests, engine_nodes: Mapping[str, str],
        pcie_submission_ps: int, tag: int = 6200,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        from simllm.backends.flow_session import FlowSessionConfig

        if type(config) is not FlowSessionConfig or config.profile != "rnic-nn":
            raise ValueError("shared cache transfer requires the ideal rnic-nn session profile")
        if type(pcie_submission_ps) is not int or pcie_submission_ps <= 0:
            raise ValueError("shared cache transfer requires positive submission delay")
        if type(tag) is not int or not 0 <= tag < 2**32:
            raise ValueError("shared cache transfer tag must be an unsigned 32-bit integer")
        if not isinstance(deployment, DisaggregatedDeploymentManifests):
            raise TypeError("shared handoff requires deployment manifests")
        deployment = deepcopy(deployment)
        deployment.validate()
        if deployment.fabric.physical_rendering_enabled:
            raise ValueError("ideal shared handoff requires logical placement without a physical topology claim")
        if deployment.placement.source != "declared" or deployment.fabric.goal_rank_mapping != "gpu-rank":
            raise ValueError("shared handoff requires declared GPU-rank placement")
        if config.node_count != len(deployment.placement.ranks):
            raise ValueError("native endpoint count disagrees with the deployment")
        if not isinstance(engine_nodes, Mapping) or any(
            type(key) is not str or not key.strip() or type(node) is not str or not node.strip()
            for key, node in engine_nodes.items()
        ):
            raise ValueError("engine bindings must contain nonblank string identities")
        nodes = {node.node_id: node for node in deployment.fabric.nodes}
        if len(set(engine_nodes.values())) != len(engine_nodes) or set(engine_nodes.values()) != set(nodes):
            raise ValueError("engine bindings must cover deployment nodes exactly once")
        ranks = deployment.placement.ranks
        self._engine_ranks: dict[str, tuple[int, ...]] = {}
        self._engine_roles: dict[str, str] = {}
        for engine, node in engine_nodes.items():
            group = tuple(rank for rank in ranks if rank.hostname == node)
            local = tuple(rank.local_rank for rank in group)
            global_ranks = tuple(rank.global_rank for rank in group)
            if local != tuple(range(len(group))) or not group:
                raise ValueError("each engine must have dense ordered local worker ranks")
            if any(rank.groups.get("tp") is None
                   or tuple(rank.groups["tp"].global_ranks) != global_ranks
                   or rank.groups["tp"].rank_in_group != rank.local_rank for rank in group):
                raise ValueError("engine deployment must contain one complete ordered tensor-parallel group")
            self._engine_ranks[engine] = global_ranks
            self._engine_roles[engine] = nodes[node].pool_role
        self._config = replace(config)
        self._command = tuple(command)
        if not self._command or any(type(item) is not str or not item for item in self._command):
            raise ValueError("native command must contain nonempty string arguments")
        if type(session_id) is not str or not session_id.strip():
            raise ValueError("native session identity must be nonblank")
        self._session_id = session_id
        self._environment = None if environment is None else dict(environment)
        self._submission_ps = pcie_submission_ps
        self._tag = tag
        self._clock: VirtualClock | None = None
        self._session = None
        self._validated = False
        self._closed = False
        self._poisoned = False
        self._abort_error: str | None = None
        self._receipts: dict[str, PendingKvHandoff] = {}
        self._receipt_values: dict[str, tuple] = {}
        self._pending: dict[str, PendingKvHandoff] = {}
        self._flow_owner: dict[int, str] = {}
        self._events: dict[int, list] = {}
        self._observed_events: list[Mapping[str, object]] = []
        self._rows: dict[int, Mapping] = {}
        self._staged: dict[str, KvHandoffJoin] = {}
        self._published: list[KvHandoffJoin] = []
        self._accepted: list[Mapping[str, object]] = []
        self._drain = None

    def _guard(self) -> None:
        if self._closed or self._poisoned:
            raise RuntimeError("shared handoff owner is closed or poisoned")
        try:
            for request, receipt in self._receipts.items():
                receipt.__post_init__()
                if astuple(receipt) != self._receipt_values[request]:
                    raise RuntimeError("accepted handoff receipt was changed")
        except BaseException:
            self._abort_after_failure()
            raise

    def _abort_after_failure(self) -> None:
        try:
            self.abort()
        except BaseException as error:  # noqa: BLE001, preserve the original failure.
            self._abort_error = str(error)

    def validate_engines(self, prefill_ids: Sequence[str], decode_ids: Sequence[str], width: int) -> None:
        self._guard()
        identities = (*prefill_ids, *decode_ids)
        if len(identities) != len(set(identities)) or set(identities) != set(self._engine_ranks):
            raise ValueError("actual engine inventory disagrees with shared handoff bindings")
        for role, engines in ((ServingPoolRole.PREFILL, prefill_ids), (ServingPoolRole.DECODE, decode_ids)):
            if not engines or any(self._engine_roles[engine] != role.value
                                  or len(self._engine_ranks[engine]) != width for engine in engines):
                raise ValueError("actual engine role or width disagrees with shared handoff placement")
        self._validated = True

    def bind(self, clock: VirtualClock, engine_local_ranks: Mapping[str, tuple[int, ...]]) -> None:
        from simllm.backends.flow_session import FlowSession

        self._guard()
        if self._clock is not None or not self._validated:
            raise RuntimeError("shared handoff must bind exactly once after engine validation")
        if not isinstance(clock, VirtualClock) or set(engine_local_ranks) != set(self._engine_ranks):
            raise ValueError("shared handoff clock or actual engine inventory is invalid")
        if any(type(local) is not tuple or any(type(rank) is not int for rank in local)
               or local != tuple(range(len(self._engine_ranks[engine])))
               for engine, local in engine_local_ranks.items()):
            raise ValueError("actual worker local ranks disagree with their global deployment binding")
        self._clock = clock
        self._session = FlowSession(self._config, self._command, session_id=self._session_id,
                                    time_origin_ps=clock.now_ps, environment=self._environment)
        try:
            self._session.open()
        except BaseException:
            self._abort_after_failure()
            raise

    def _bound(self) -> None:
        self._guard()
        if self._session is None or self._clock is None:
            raise RuntimeError("shared handoff owner has not been bound")

    def _check_pending(self, pending: Sequence[PendingKvHandoff]) -> None:
        try:
            self._bound()
            values = tuple(pending)
            if (any(type(value) is not PendingKvHandoff for value in values)
                    or len(values) != len(self._pending)
                    or len({value.request_id for value in values}) != len(values)
                    or any(self._pending.get(value.request_id) is not value for value in values)):
                raise RuntimeError("pending handoff inventory is missing, foreign, forged or already published")
        except BaseException:
            self._abort_after_failure()
            raise

    def submit(self, *, request_id: str, source_engine_id: str, destination_engine_id: str,
               kv_bytes: int, submitted_at_ps: int) -> PendingKvHandoff:
        self._bound()
        if type(request_id) is not str or not request_id.strip() or "\x00" in request_id or len(request_id.encode()) > 4096:
            raise ValueError("shared handoff request identity is invalid")
        if request_id in self._receipts:
            raise ValueError("duplicate shared handoff request")
        if type(submitted_at_ps) is not int or submitted_at_ps != self._clock.now_ps:
            raise ValueError("handoff submission must equal the public completion time")
        if (self._engine_roles.get(source_engine_id) != ServingPoolRole.PREFILL.value
                or self._engine_roles.get(destination_engine_id) != ServingPoolRole.DECODE.value):
            raise ValueError("handoff must bind an actual producer and consumer engine")
        sources, destinations = self._engine_ranks[source_engine_id], self._engine_ranks[destination_engine_id]
        if len(sources) != len(destinations) or type(kv_bytes) is not int or kv_bytes < len(sources):
            raise ValueError("handoff needs equal engine widths and positive bytes per shard")
        eligible = submitted_at_ps + self._submission_ps
        if eligible > self._session.max_time_ps:
            raise ValueError("handoff eligibility exceeds the native session budget")
        quotient, remainder = divmod(kv_bytes, len(sources))
        operation = "kv/" + quote(request_id, safe="")
        fields = [{"execution_id": self._session_id, "operation_id": operation,
                   "flow_id": operation + f"/shard/{index}", "source": source, "destination": destination,
                   "payload_bytes": quotient + (index < remainder), "tag": self._tag, "eligible_at_ps": eligible}
                  for index, (source, destination) in enumerate(zip(sources, destinations, strict=True))]
        accepted = []
        try:
            for values in fields:
                sequence = self._session.inject(**values)
                accepted.append(sequence)
                self._accepted.append(MappingProxyType({**values, "sequence": sequence, "request_id": request_id}))
                self._flow_owner[sequence] = request_id
                self._events[sequence] = []
            receipt = PendingKvHandoff(request_id, source_engine_id, destination_engine_id,
                                       kv_bytes, submitted_at_ps, eligible, tuple(accepted))
            self._receipts[request_id] = receipt
            self._receipt_values[request_id] = astuple(receipt)
            self._pending[request_id] = receipt
            return receipt
        except BaseException:
            self._abort_after_failure()
            raise

    def _progress(self, through_ps: int | None) -> int | None:
        if not self._session.pending_sequences:
            return None
        update = self._session.await_completion(through_ps=through_ps)
        for event in update.events:
            sequence = event["sequence"]
            if sequence not in self._flow_owner:
                raise RuntimeError("native lifecycle belongs to a foreign handoff")
            self._events[sequence].append(event)
            self._observed_events.append(event)
        for row in update.completion_rows:
            sequence = row["sequence"]
            if sequence not in self._flow_owner or sequence in self._rows:
                raise RuntimeError("native handoff completion is foreign or duplicated")
            if row["completion_time_ps"] < self._clock.now_ps:
                raise RuntimeError("native handoff completion arrived after its public time")
            self._rows[sequence] = row
            request = self._flow_owner[sequence]
            receipt = self._pending[request]
            if all(value in self._rows for value in receipt.sequences):
                if request in self._staged:
                    raise RuntimeError("native handoff join was staged twice")
                self._staged[request] = KvHandoffJoin(receipt, tuple(
                    KvHandoffShardCompletion(self._rows[value], tuple(self._events[value]))
                    for value in receipt.sequences))
        return update.event_time_ps if update.reason == "completion" else None

    def progress(self, pending: Sequence[PendingKvHandoff], *, through_ps: int | None) -> int | None:
        self._check_pending(pending)
        if self._staged:
            raise RuntimeError("publish discovered handoff joins before requesting later progress")
        if through_ps is not None and (type(through_ps) is not int or through_ps < self._clock.now_ps):
            raise ValueError("native lookahead cannot precede the public clock")
        try:
            return self._progress(through_ps)
        except BaseException:
            self._abort_after_failure()
            raise

    def complete_due(self, pending: Sequence[PendingKvHandoff]) -> tuple[KvHandoffJoin, ...]:
        self._check_pending(pending)
        now = self._clock.now_ps
        if any(join.completed_at_ps < now for join in self._staged.values()):
            self._abort_after_failure()
            raise RuntimeError("handoff join publication missed its exact completion")
        # Lookahead may have found a future join; it remains private until C.
        if any(join.completed_at_ps > now for join in self._staged.values()):
            return ()
        floor = self._session.ordinary_injection_floor_ps
        if floor is not None and floor > now:
            return ()
        try:
            while self._session.pending_sequences:
                if self._progress(now) is None:
                    break
            ready = tuple(join for request in self._pending
                          if (join := self._staged.get(request)) is not None)
            if any(join.completed_at_ps != now for join in ready):
                raise RuntimeError("handoff join escaped its exact public completion")
            for join in ready:
                del self._staged[join.request_id]
                del self._pending[join.request_id]
                self._published.append(join)
            return ready
        except BaseException:
            self._abort_after_failure()
            raise

    @property
    def accepted_shards(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._accepted)

    @property
    def joins(self) -> tuple[KvHandoffJoin, ...]:
        return tuple(self._published)

    @property
    def native_events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._observed_events)

    @property
    def native_rows(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._rows[sequence] for sequence in sorted(self._rows))

    @property
    def transcript(self) -> tuple[tuple[str, bytes], ...]:
        return () if self._session is None else self._session.transcript

    @property
    def stderr(self) -> bytes:
        return b"" if self._session is None else self._session.stderr

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def close(self) -> None:
        if self._closed:
            return
        self._guard()
        if self._pending or self._staged:
            raise RuntimeError("cannot close a shared handoff owner with unpublished work")
        if self._session is not None:
            self._drain = self._session.close()
        self._closed = True

    def abort(self) -> None:
        self._poisoned = True
        if self._session is not None:
            self._session.abort()


__all__ = ["SharedKvHandoffRuntime"]
