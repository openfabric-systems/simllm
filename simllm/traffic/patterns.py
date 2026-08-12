"""Communication patterns rendered as GOAL fragments.

Each function appends operations to a :class:`~simllm.goal.GoalTrace` and
returns per-rank labels so callers can chain phases (`after` in, completion
labels out). Semantic collectives are expanded into the point-to-point
algorithm actually used, so the network simulator sees the real chunked
traffic pattern rather than an abstract op.

Conventions: `after` maps rank to a label the phase must wait for on that
rank (missing rank means no dependency); the returned dict maps rank to the
label of that rank's last operation in the phase.
"""

from __future__ import annotations

from collections.abc import Sequence

from simllm.goal import (
    GoalDependencyKind,
    GoalDependencyProvenance,
    GoalMessage,
    GoalTrace,
)


def _chain(
    trace: GoalTrace,
    rank: int,
    op_label: str,
    after: dict[int, str] | None,
    after_provenance: dict[int, GoalDependencyProvenance] | None = None,
) -> None:
    if after and rank in after:
        trace.rank(rank).requires(
            op_label,
            after[rank],
            provenance=None if after_provenance is None else after_provenance.get(rank),
        )


def scatter(
    trace: GoalTrace,
    root: int,
    workers: list[int],
    size_bytes: int,
    tag: int,
    after: dict[int, str] | None = None,
) -> dict[int, str]:
    """Root sends one message to every worker; workers complete on receive."""
    done: dict[int, str] = {}
    for w in workers:
        tx = trace.rank(root).send(size_bytes, to=w, tag=tag)
        _chain(trace, root, tx, after)
        rx = trace.rank(w).recv(size_bytes, source=root, tag=tag)
        _chain(trace, w, rx, after)
        done[w] = rx
        done[root] = tx
    return done


def gather(
    trace: GoalTrace,
    root: int,
    workers: list[int],
    size_bytes: int,
    tag: int,
    after: dict[int, str] | None = None,
) -> dict[int, str]:
    """Every worker sends one message to root; root completes on last receive."""
    done: dict[int, str] = {}
    last_rx = None
    for w in workers:
        tx = trace.rank(w).send(size_bytes, to=root, tag=tag)
        _chain(trace, w, tx, after)
        done[w] = tx
        rx = trace.rank(root).recv(size_bytes, source=w, tag=tag)
        _chain(trace, root, rx, after)
        last_rx = rx
    if last_rx is not None:
        done[root] = last_rx
    return done


def ring_allreduce(
    trace: GoalTrace,
    ranks: list[int],
    size_bytes: int,
    base_tag: int,
    after: dict[int, str] | None = None,
    *,
    operation_id: str | None = None,
    after_provenance: dict[int, GoalDependencyProvenance] | None = None,
    exact_frontier: bool = False,
) -> dict[int, str]:
    """Ring allreduce: reduce-scatter then allgather, 2*(W-1) neighbor rounds.

    Each round every rank sends one chunk of size ``size_bytes / W`` to its
    ring successor and receives one from its predecessor; a rank's round
    starts only when its previous round is complete (send and recv).
    """
    world = len(ranks)
    if world < 2:
        raise ValueError("ring allreduce needs at least 2 ranks")
    chunk = max(1, size_bytes // world)
    prev_done: dict[int, str] = dict(after or {})
    for round_index in range(2 * (world - 1)):
        tag = base_tag + round_index
        round_done: dict[int, str] = {}
        round_labels: dict[int, tuple[str, str]] = {}
        for i, r in enumerate(ranks):
            succ = ranks[(i + 1) % world]
            pred = ranks[(i - 1) % world]
            tx = trace.rank(r).send(
                chunk,
                to=succ,
                tag=tag,
                operation_id=operation_id,
            )
            rx = trace.rank(r).recv(
                chunk,
                source=pred,
                tag=tag,
                operation_id=operation_id,
            )
            round_labels[r] = (tx, rx)
        for i, r in enumerate(ranks):
            succ = ranks[(i + 1) % world]
            tx, rx = round_labels[r]
            trace.record_message(
                GoalMessage(
                    operation_id=operation_id,
                    source_rank=r,
                    destination_rank=succ,
                    payload_bytes=chunk,
                    tag=tag,
                    send_label=tx,
                    receive_label=round_labels[succ][1],
                )
            )
            if r in prev_done:
                if round_index == 0:
                    provenance = (
                        None if after_provenance is None else after_provenance.get(r)
                    )
                elif operation_id is None:
                    provenance = None
                else:
                    provenance = GoalDependencyProvenance(
                        GoalDependencyKind.COLLECTIVE_INTERNAL,
                        operation_id,
                    )
                trace.rank(r).requires(tx, prev_done[r], provenance=provenance)
                trace.rank(r).requires(rx, prev_done[r], provenance=provenance)
            if exact_frontier:
                if operation_id is None:
                    raise ValueError("exact ring frontier requires operation_id")
                join = trace.rank(r).calc(0, operation_id=operation_id)
                internal = GoalDependencyProvenance(
                    GoalDependencyKind.COLLECTIVE_INTERNAL,
                    operation_id,
                )
                trace.rank(r).requires(join, tx, provenance=internal)
                trace.rank(r).requires(join, rx, provenance=internal)
                round_done[r] = join
            else:
                # Compatibility rendering selects the receive as its syntactic
                # frontier. Exact graph projection uses the join above.
                round_done[r] = rx
        prev_done = round_done
    return prev_done


def pairwise_all_to_allv(
    trace: GoalTrace,
    ranks: list[int],
    send_bytes: dict[tuple[int, int], int],
    tag: int,
    after: dict[int, str] | None = None,
    *,
    operation_id: str | None = None,
    after_provenance: dict[int, GoalDependencyProvenance] | None = None,
    exact_frontier: bool = False,
    request_send_bytes: dict[tuple[int, int], tuple[tuple[str, int], ...]] | None = None,
) -> dict[int, str]:
    """Direct pairwise exchange: rank s sends ``send_bytes[(s, d)]`` to d.

    Zero or missing entries send nothing. Completion label per rank is its
    last receive, or its first send when it receives nothing at all: a
    source-only rank hands its successor the head of its own send chain, not
    the tail. ``render_collective_plan`` documents and implements the same
    compatibility rule, and the two statements are kept in agreement
    deliberately. A last-send frontier for source-only ranks would be a
    separate, separately frozen model decision because it moves accepted
    timing. When an exact semantic frontier is requested, a rank with no
    incident message receives a zero-time completion point instead of
    disappearing from the collective.
    """
    positive_pairs = {
        (source, destination)
        for (source, destination), size in send_bytes.items()
        if size > 0 and source != destination
    }
    if request_send_bytes is not None and set(request_send_bytes) != positive_pairs:
        missing = sorted(positive_pairs - set(request_send_bytes))
        extra = sorted(set(request_send_bytes) - positive_pairs)
        raise ValueError(
            "request_send_bytes must cover exactly the positive physical pairs; "
            f"missing={missing}, extra={extra}"
        )

    done: dict[int, str] = {}
    incident: dict[int, list[str]] = {rank: [] for rank in ranks}
    for s in ranks:
        for d in ranks:
            size = send_bytes.get((s, d), 0)
            if size <= 0 or s == d:
                continue
            tx = trace.rank(s).send(
                size,
                to=d,
                tag=tag,
                operation_id=operation_id,
            )
            _chain(trace, s, tx, after, after_provenance)
            done.setdefault(s, tx)
            incident[s].append(tx)
            rx = trace.rank(d).recv(
                size,
                source=s,
                tag=tag,
                operation_id=operation_id,
            )
            _chain(trace, d, rx, after, after_provenance)
            done[d] = rx
            incident[d].append(rx)
            trace.record_message(
                GoalMessage(
                    operation_id=operation_id,
                    source_rank=s,
                    destination_rank=d,
                    payload_bytes=size,
                    tag=tag,
                    send_label=tx,
                    receive_label=rx,
                    request_payload_bytes=(
                        () if request_send_bytes is None else request_send_bytes[(s, d)]
                    ),
                )
            )
    if exact_frontier:
        if operation_id is None:
            raise ValueError("exact pairwise frontier requires operation_id")
        internal = GoalDependencyProvenance(
            GoalDependencyKind.COLLECTIVE_INTERNAL,
            operation_id,
        )
        for rank in ranks:
            labels = incident[rank]
            if not labels:
                if after and rank in after:
                    done[rank] = after[rank]
                else:
                    done[rank] = trace.rank(rank).calc(
                        0,
                        operation_id=operation_id,
                    )
                continue
            join = trace.rank(rank).calc(0, operation_id=operation_id)
            for label in labels:
                trace.rank(rank).requires(join, label, provenance=internal)
            done[rank] = join
    return done


def ordered_pairwise_messages(
    trace: GoalTrace,
    ranks: list[int],
    messages: Sequence[tuple[str, int, int, int]],
    tag: int,
    after: dict[int, str] | None = None,
    *,
    operation_id: str | None = None,
    after_provenance: dict[int, GoalDependencyProvenance] | None = None,
) -> dict[int, str]:
    """Emit request-attributed messages in their supplied issue order.

    Each row is ``(request_id, source, destination, payload_bytes)``. Tuple
    order is authoritative. Sends from one source are connected with
    ``irequires`` so they enter that source's queue in order without waiting
    for the previous message to finish. Different sources remain unordered.
    The returned frontier joins every incident send and receive before the
    caller advances to the next phase.
    """

    if operation_id is None:
        raise ValueError("ordered messages require operation_id for their frontier")
    rank_set = set(ranks)
    if not ranks or len(rank_set) != len(ranks):
        raise ValueError("ranks must be nonempty and distinct")
    rows = tuple(messages)
    if not rows:
        raise ValueError("messages must not be empty")

    for index, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 4:
            raise TypeError(f"messages[{index}] must be a four-item tuple")
        request_id, source, destination, payload_bytes = row
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(f"messages[{index}][0] must be a nonblank string")
        if source not in rank_set or destination not in rank_set:
            raise ValueError(f"messages[{index}] uses a rank outside the group")
        if source == destination:
            raise ValueError(f"messages[{index}] is a self-pair")
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int):
            raise TypeError(f"messages[{index}][3] must be an integer")
        if payload_bytes <= 0:
            raise ValueError(f"messages[{index}][3] must be positive")

    done: dict[int, str] = {}
    incident: dict[int, list[str]] = {rank: [] for rank in ranks}
    send_tails: dict[int, str] = {}
    internal = GoalDependencyProvenance(
        GoalDependencyKind.COLLECTIVE_INTERNAL,
        operation_id,
    )
    for index, row in enumerate(rows):
        request_id, source, destination, payload_bytes = row

        tx = trace.rank(source).send(
            payload_bytes,
            to=destination,
            tag=tag,
            operation_id=operation_id,
        )
        if source in send_tails:
            trace.rank(source).irequires(
                tx,
                send_tails[source],
                provenance=internal,
            )
        else:
            _chain(trace, source, tx, after, after_provenance)
        send_tails[source] = tx
        incident[source].append(tx)

        rx = trace.rank(destination).recv(
            payload_bytes,
            source=source,
            tag=tag,
            operation_id=operation_id,
        )
        _chain(trace, destination, rx, after, after_provenance)
        incident[destination].append(rx)
        trace.record_message(
            GoalMessage(
                operation_id=operation_id,
                source_rank=source,
                destination_rank=destination,
                payload_bytes=payload_bytes,
                tag=tag,
                send_label=tx,
                receive_label=rx,
                request_payload_bytes=((request_id, payload_bytes),),
            )
        )

    for rank, labels in incident.items():
        if not labels:
            continue
        join = trace.rank(rank).calc(0, operation_id=operation_id)
        for label in labels:
            trace.rank(rank).requires(join, label, provenance=internal)
        done[rank] = join
    return done


def binomial_broadcast(
    trace: GoalTrace,
    root: int,
    ranks: list[int],
    size_bytes: int,
    tag: int,
    after: dict[int, str] | None = None,
) -> dict[int, str]:
    """Binomial-tree broadcast in ceil(log2(W)) rounds.

    Round k: every rank that already holds the data sends to the rank
    ``2^k`` positions away (in root-rotated order), doubling the holders.
    """
    order = [root] + [r for r in ranks if r != root]
    done: dict[int, str] = {}
    if after and root in after:
        done[root] = after[root]
    holders = 1
    while holders < len(order):
        for i in range(min(holders, len(order) - holders)):
            src, dst = order[i], order[i + holders]
            tx = trace.rank(src).send(size_bytes, to=dst, tag=tag)
            if src in done:
                trace.rank(src).requires(tx, done[src])
            done[src] = tx
            rx = trace.rank(dst).recv(size_bytes, source=src, tag=tag)
            if after and dst in after:
                trace.rank(dst).requires(rx, after[dst])
            done[dst] = rx
        holders *= 2
    return done
