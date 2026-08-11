"""Project composed native RNIC observations into the core runtime seam.

The external composed producer owns every mutable WQE lifecycle. This module
validates its immutable Tier A observation rows and adapts one selected row to
``NativeRnicSession``. Transaction state below tracks projection consumption
only. It never advances a WQE, network port, queue, or simulator clock.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.core.runtime import (
    SemanticWqeSubmission,
    WqeLifecycleProjection,
)

NATIVE_AUTHORITY = "SimllmNativeRnicSession"
TIER_A_EXPECTATION_SCHEMA = "simllm-rnic-tier-a-expectations-v1"
TIER_A_OBSERVATION_SCHEMA = "simllm-rnic-tier-a-observations-v1"


class ComposedRnicObservationError(ValueError):
    """A composed producer row cannot be projected without disagreement."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComposedRnicObservationError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be an array")
    return value


def _integer(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{name} keys differ: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


@dataclass(frozen=True)
class ComposedWqeObservation:
    """One validated native WQE and htsim terminal projection."""

    ordinal: int
    native_wqe_id: int
    eligible_at_ps: int
    network_started_at_ps: int
    network_finished_at_ps: int
    completed_at_ps: int

    def __post_init__(self) -> None:
        _require(self.ordinal >= 0, "composed WQE ordinal must be nonnegative")
        _require(self.native_wqe_id > 0, "composed WQE ID must be positive")
        _require(
            0
            <= self.eligible_at_ps
            <= self.network_started_at_ps
            <= self.network_finished_at_ps
            <= self.completed_at_ps,
            "composed WQE timestamps are not monotonic",
        )


@dataclass(frozen=True)
class ComposedRnicCell:
    """One exact structural fixture cell emitted by the composed producer."""

    payload_bytes: int
    link_rate_gbps: int
    doorbell_service_ps: int
    wqes: tuple[ComposedWqeObservation, ...]
    jct_ps: int

    def __post_init__(self) -> None:
        _require(self.payload_bytes > 0, "composed payload must be positive")
        _require(self.link_rate_gbps > 0, "composed link rate must be positive")
        _require(
            self.doorbell_service_ps >= 0,
            "composed doorbell service must be nonnegative",
        )
        _require(len(self.wqes) in {1, 2}, "composed cell must contain one or two WQEs")
        _require(
            tuple(wqe.ordinal for wqe in self.wqes) == tuple(range(len(self.wqes))),
            "composed WQE ordinals must be contiguous",
        )
        _require(
            all(
                wqe.eligible_at_ps == self.doorbell_service_ps
                for wqe in self.wqes
            ),
            "composed WQE eligibility disagrees with doorbell service",
        )
        _require(
            self.jct_ps == max(wqe.completed_at_ps for wqe in self.wqes),
            "composed JCT disagrees with WQE completion",
        )

    @property
    def wqe_count(self) -> int:
        return len(self.wqes)


def _parse_wqe(value: Any, name: str) -> ComposedWqeObservation:
    row = _object(value, name)
    _exact_keys(
        row,
        {
            "ordinal",
            "wqe_id",
            "eligible_at_ps",
            "network_accepted_at_ps",
            "port_tx_at_ps",
            "terminal_kind",
            "terminal_at_ps",
            "cqe_status",
            "cqe_visible_at_ps",
            "polled_at_ps",
        },
        name,
    )
    ordinal = _integer(row["ordinal"], f"{name}.ordinal")
    native_wqe_id = _integer(row["wqe_id"], f"{name}.wqe_id")
    eligible = _integer(row["eligible_at_ps"], f"{name}.eligible_at_ps")
    accepted = _integer(
        row["network_accepted_at_ps"],
        f"{name}.network_accepted_at_ps",
    )
    started = _integer(row["port_tx_at_ps"], f"{name}.port_tx_at_ps")
    terminal = _integer(row["terminal_at_ps"], f"{name}.terminal_at_ps")
    cqe_visible = _integer(
        row["cqe_visible_at_ps"],
        f"{name}.cqe_visible_at_ps",
    )
    completed = _integer(row["polled_at_ps"], f"{name}.polled_at_ps")
    _require(native_wqe_id > 0, f"{name} has a zero native WQE ID")
    _require(
        row["terminal_kind"] == "delivered" and row["cqe_status"] == "success",
        f"{name} is not a successful native completion",
    )
    _require(
        0 <= eligible <= accepted <= started <= terminal <= cqe_visible <= completed,
        f"{name} timestamps are not monotonic",
    )
    _require(
        accepted == started,
        f"{name} does not use the frozen zero-admission fixture",
    )
    return ComposedWqeObservation(
        ordinal=ordinal,
        native_wqe_id=native_wqe_id,
        eligible_at_ps=eligible,
        network_started_at_ps=started,
        network_finished_at_ps=terminal,
        completed_at_ps=completed,
    )


def _parse_cell(value: Any, name: str, expected_wqes: int) -> ComposedRnicCell:
    cell = _object(value, name)
    _exact_keys(
        cell,
        {
            "payload_bytes",
            "link_rate_gbps",
            "doorbell_service_ps",
            "authority",
            "device",
            "port",
            "wqes",
            "cqe_order",
            "jct_ps",
        },
        name,
    )
    payload = _integer(cell["payload_bytes"], f"{name}.payload_bytes")
    rate = _integer(cell["link_rate_gbps"], f"{name}.link_rate_gbps")
    doorbell = _integer(
        cell["doorbell_service_ps"],
        f"{name}.doorbell_service_ps",
    )
    jct = _integer(cell["jct_ps"], f"{name}.jct_ps")
    _require(payload > 0 and rate > 0, f"{name} has a nonpositive fixture input")
    _require(doorbell >= 0, f"{name} has a negative doorbell service")

    authority = _object(cell["authority"], f"{name}.authority")
    _require(
        authority
        == {
            "mode": "structural",
            "native_session_constructed": 1,
            "native_posts": expected_wqes,
            "legacy_ledger_constructed": 0,
            "legacy_posts": 0,
            "legacy_mutations": 0,
        },
        f"{name} violates sole native authority",
    )
    device = _object(cell["device"], f"{name}.device")
    counters = _object(device.get("counters"), f"{name}.device.counters")
    _require(
        counters
        == {
            "posted_wqes": expected_wqes,
            "network_accepted": expected_wqes,
            "network_delivered": expected_wqes,
            "network_dropped": 0,
        },
        f"{name} native counters do not conserve WQEs",
    )
    _require(
        device.get("has_pending_physical_work") is False
        and device.get("occupied_sq_entries") == 0
        and device.get("completion_queue_depth") == 0
        and device.get("unpublished_wqes") == 0
        and device.get("fatal") is False,
        f"{name} did not quiesce cleanly",
    )
    port = _object(cell["port"], f"{name}.port")
    issued = _array(port.get("issued"), f"{name}.port.issued")
    terminals = _array(port.get("terminals"), f"{name}.port.terminals")
    live_tokens = _array(port.get("live_tokens"), f"{name}.port.live_tokens")
    _require(
        len(issued) == expected_wqes
        and len(terminals) == expected_wqes
        and not live_tokens,
        f"{name} port token ledger is not quiescent",
    )

    raw_wqes = _array(cell["wqes"], f"{name}.wqes")
    _require(len(raw_wqes) == expected_wqes, f"{name} has the wrong WQE count")
    wqes = tuple(
        _parse_wqe(row, f"{name}.wqes[{index}]")
        for index, row in enumerate(raw_wqes)
    )
    _require(
        tuple(wqe.ordinal for wqe in wqes) == tuple(range(expected_wqes)),
        f"{name} WQE ordinals are not contiguous",
    )
    native_ids = tuple(wqe.native_wqe_id for wqe in wqes)
    _require(len(set(native_ids)) == len(native_ids), f"{name} repeats a WQE ID")
    _require(
        tuple(_array(cell["cqe_order"], f"{name}.cqe_order")) == native_ids,
        f"{name} CQE order disagrees with native WQE order",
    )
    _require(
        all(wqe.eligible_at_ps == doorbell for wqe in wqes),
        f"{name} doorbell completion disagrees with D",
    )
    _require(
        jct == max(wqe.completed_at_ps for wqe in wqes),
        f"{name} JCT disagrees with native CQ polling",
    )
    return ComposedRnicCell(payload, rate, doorbell, wqes, jct)


@dataclass(frozen=True)
class ComposedRnicObservations:
    """Validated lookup over the composed htsim Tier A structural matrix."""

    single_wqe: dict[tuple[int, int, int], ComposedRnicCell]
    fifo: dict[tuple[int, int], ComposedRnicCell]

    @classmethod
    def from_json(cls, value: Any) -> ComposedRnicObservations:
        root = _object(value, "composed observations")
        _require(
            root.get("schema") == TIER_A_OBSERVATION_SCHEMA,
            "composed observations use the wrong schema",
        )
        _require(root.get("factory") == "htsim", "composed factory is not htsim")
        single_rows = _array(root.get("single_wqe"), "single_wqe")
        fifo_rows = _array(root.get("fifo"), "fifo")
        single: dict[tuple[int, int, int], ComposedRnicCell] = {}
        for index, raw in enumerate(single_rows):
            cell = _parse_cell(raw, f"single_wqe[{index}]", 1)
            key = (
                cell.payload_bytes,
                cell.link_rate_gbps,
                cell.doorbell_service_ps,
            )
            _require(key not in single, f"single_wqe repeats cell {key}")
            single[key] = cell
        fifo: dict[tuple[int, int], ComposedRnicCell] = {}
        for index, raw in enumerate(fifo_rows):
            cell = _parse_cell(raw, f"fifo[{index}]", 2)
            _require(cell.payload_bytes == 4096, "FIFO payload is not 4 KiB")
            key = (cell.link_rate_gbps, cell.doorbell_service_ps)
            _require(key not in fifo, f"fifo repeats cell {key}")
            fifo[key] = cell

        expected_single = {
            (payload, rate, doorbell)
            for payload in (4096, 1_048_576)
            for rate in (200, 400)
            for doorbell in (0, 1000)
        }
        expected_fifo = {
            (rate, doorbell) for rate in (200, 400) for doorbell in (0, 1000)
        }
        _require(set(single) == expected_single, "single-WQE matrix is incomplete")
        _require(set(fifo) == expected_fifo, "FIFO matrix is incomplete")
        return cls(single_wqe=single, fifo=fifo)

    @classmethod
    def read(cls, path: str | Path) -> ComposedRnicObservations:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ComposedRnicObservationError(
                f"cannot read composed observations {source}: {error}"
            ) from error
        return cls.from_json(value)


def invoke_composed_tier_a_producer(
    producer: str | Path,
    expectations: str | Path,
    observations: str | Path,
) -> list[str]:
    """Invoke the frozen Tier A port-factory producer and return its argv."""

    producer_path = Path(producer).resolve(strict=True)
    expectations_path = Path(expectations).resolve(strict=True)
    observations_path = Path(observations).resolve(strict=False)
    _require(
        not observations_path.exists(),
        f"composed observation file already exists: {observations_path}",
    )
    command = [
        str(producer_path),
        "--factory",
        "htsim",
        "--expectations",
        str(expectations_path),
        "--observations",
        str(observations_path),
    ]
    subprocess.run(command, check=True)
    _require(
        observations_path.is_file(),
        "composed producer returned without publishing observations",
    )
    return command


class ComposedRnicSession:
    """Read-only native timing adapter satisfying ``NativeRnicSession``."""

    authority_name = NATIVE_AUTHORITY

    def __init__(self, cell: ComposedRnicCell, *, session_id: str) -> None:
        if not isinstance(cell, ComposedRnicCell):
            raise TypeError("cell must be a ComposedRnicCell")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a nonblank string")
        self.cell = cell
        self.session_id = session_id
        self._committed_transactions = 0
        self._committed_wqes = 0
        self._active: _ComposedRnicTransaction | None = None

    @property
    def committed_transactions(self) -> int:
        return self._committed_transactions

    @property
    def committed_wqes(self) -> int:
        return self._committed_wqes

    def begin_transaction(self) -> _ComposedRnicTransaction:
        if self._active is not None:
            raise RuntimeError("composed RNIC session already has an active transaction")
        transaction = _ComposedRnicTransaction(
            self,
            transaction_index=self._committed_transactions,
            sequence_base=self._committed_wqes,
        )
        self._active = transaction
        return transaction

    def _commit(self, transaction: _ComposedRnicTransaction) -> None:
        if self._active is not transaction:
            raise RuntimeError("composed RNIC transaction is not active")
        self._committed_transactions += 1
        self._committed_wqes += self.cell.wqe_count
        self._active = None

    def _abort(self, transaction: _ComposedRnicTransaction) -> None:
        if self._active is transaction:
            self._active = None


class _ComposedRnicTransaction:
    """Atomic consumption of one immutable native observation cell."""

    authority_name = NATIVE_AUTHORITY

    def __init__(
        self,
        session: ComposedRnicSession,
        *,
        transaction_index: int,
        sequence_base: int,
    ) -> None:
        self._session = session
        self._transaction_index = transaction_index
        self._sequence_base = sequence_base
        self._projections: list[WqeLifecycleProjection] = []
        self._prepared = False
        self._finished = False

    @property
    def random_draw_count(self) -> int:
        return 0

    def submit(self, submission: SemanticWqeSubmission) -> WqeLifecycleProjection:
        if self._finished or self._prepared:
            raise RuntimeError("composed RNIC transaction no longer accepts submissions")
        if not isinstance(submission, SemanticWqeSubmission):
            raise TypeError("submission must be a SemanticWqeSubmission")
        ordinal = len(self._projections)
        if ordinal >= self._session.cell.wqe_count:
            raise ValueError("graph submitted more WQEs than the selected native cell")
        if submission.payload_bytes != self._session.cell.payload_bytes:
            raise ValueError("semantic payload disagrees with the selected native cell")

        observed = self._session.cell.wqes[ordinal]
        offset = submission.eligible_at_ps
        doorbell_started = offset
        doorbell_completed = offset + observed.eligible_at_ps
        network_eligible = doorbell_completed
        network_started = offset + observed.network_started_at_ps
        network_finished = offset + observed.network_finished_at_ps
        completed = offset + observed.completed_at_ps
        escaped_execution = submission.execution_id.replace(":", "%3A")
        escaped_operation = submission.operation_id.replace(":", "%3A")
        endpoint = f"{self._session.session_id}:rank-{submission.source_rank}"
        sequence = self._sequence_base + ordinal + 1
        projection = WqeLifecycleProjection(
            authority=self.authority_name,
            execution_id=submission.execution_id,
            operation_id=submission.operation_id,
            wqe_id=(
                f"native-wqe:{escaped_execution}:{escaped_operation}:"
                f"{submission.extent_index}"
            ),
            native_wqe_id=(
                f"{self._session.session_id}:transaction-{self._transaction_index}:"
                f"wqe-{observed.native_wqe_id}"
            ),
            sq_id=f"{endpoint}:sq",
            rq_id=(
                f"{self._session.session_id}:rank-{submission.destination_rank}:rq"
            ),
            cq_id=f"{endpoint}:cq",
            qp_id=f"{endpoint}:qp",
            rnic_id=f"{endpoint}:rnic",
            source_rank=submission.source_rank,
            destination_rank=submission.destination_rank,
            payload_bytes=submission.payload_bytes,
            goal_tag=submission.goal_tag,
            extent_index=submission.extent_index,
            sq_post_sequence=sequence,
            cq_post_sequence=sequence,
            submitted_at_ps=submission.submitted_at_ps,
            eligible_at_ps=submission.eligible_at_ps,
            started_at_ps=network_started,
            finished_at_ps=network_finished,
            completed_at_ps=completed,
            channel_id=submission.channel_id,
            nccl_command_id=submission.nccl_command_id,
            doorbell_started_at_ps=doorbell_started,
            doorbell_completed_at_ps=doorbell_completed,
            network_eligible_at_ps=network_eligible,
            network_started_at_ps=network_started,
            network_finished_at_ps=network_finished,
        )
        self._projections.append(projection)
        return projection

    def prepare(self) -> None:
        if self._finished:
            raise RuntimeError("composed RNIC transaction is already finished")
        if len(self._projections) != self._session.cell.wqe_count:
            raise ValueError("graph did not consume every WQE in the selected native cell")
        self._prepared = True

    def commit(self) -> None:
        if self._finished:
            raise RuntimeError("composed RNIC transaction is already finished")
        if not self._prepared:
            raise RuntimeError("composed RNIC transaction must prepare before commit")
        self._session._commit(self)
        self._finished = True

    def abort(self) -> None:
        if self._finished:
            return
        self._session._abort(self)
        self._finished = True


__all__ = [
    "NATIVE_AUTHORITY",
    "ComposedRnicCell",
    "ComposedRnicObservationError",
    "ComposedRnicObservations",
    "ComposedRnicSession",
    "ComposedWqeObservation",
    "invoke_composed_tier_a_producer",
]
