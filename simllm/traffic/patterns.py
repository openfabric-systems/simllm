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

from simllm.goal import GoalMessage, GoalTrace


def _chain(trace: GoalTrace, rank: int, op_label: str, after: dict[int, str] | None) -> None:
    if after and rank in after:
        trace.rank(rank).requires(op_label, after[rank])


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
        for i, r in enumerate(ranks):
            succ = ranks[(i + 1) % world]
            pred = ranks[(i - 1) % world]
            tx = trace.rank(r).send(chunk, to=succ, tag=tag)
            rx = trace.rank(r).recv(chunk, source=pred, tag=tag)
            if r in prev_done:
                trace.rank(r).requires(tx, prev_done[r])
                trace.rank(r).requires(rx, prev_done[r])
            # the round is complete on a rank when its recv landed; the next
            # round's send also waits on it, which chains the pipeline
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
    request_send_bytes: dict[tuple[int, int], tuple[tuple[str, int], ...]] | None = None,
) -> dict[int, str]:
    """Direct pairwise exchange: rank s sends ``send_bytes[(s, d)]`` to d.

    Zero or missing entries send nothing. Completion label per rank is its
    last receive (or last send for ranks that receive nothing).
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
    for s in ranks:
        for d in ranks:
            size = send_bytes.get((s, d), 0)
            if size <= 0 or s == d:
                continue
            tx = trace.rank(s).send(size, to=d, tag=tag)
            _chain(trace, s, tx, after)
            done.setdefault(s, tx)
            rx = trace.rank(d).recv(size, source=s, tag=tag)
            _chain(trace, d, rx, after)
            done[d] = rx
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
