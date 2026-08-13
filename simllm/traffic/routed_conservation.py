"""Independent conservation guard for routed-MoE byte tables.

The guard exists because every earlier byte check on the routed path compared
the renderer against itself, which is how an 8x source-replication defect
survived for the project's entire history. Independence here has one concrete
meaning: nothing this module consumes is derived from the per-token routing
walk that builds the byte table it inspects.

The inputs are

- :class:`RoutedTokenOwnership`, the adapter's source-observed per-request
  token counts plus the one rank declared to own them
  (``RoutedMoeSupply.engine_rank``),
- the model geometry (``top_k``, ``num_layers``, and one hidden vector's
  bytes),
- the expert-parallel group,

and the object under test is the traffic planner's per-layer directed pair
table. A defect inside the routing walk therefore cannot hide inside the
check.

Five rules hold for every routed representation, including the uniform
destination approximation:

``source-attribution``
    every dispatch pair leaves the one declared owner rank and every combine
    pair arrives at it, because the owner sends its tokens out to the expert
    owners and receives the pre-reduced results back;
``destination-legality``
    both endpoints are distinct members of the expert-parallel group;
``owner-egress``
    the phase-agnostic form of the same idea: every directed byte has the
    owner at exactly one endpoint, and the step's owner egress equals its
    owner ingress. It survives a producer that mislabels the two phases,
    which ``source-attribution`` does not;
``transpose-symmetry``
    each layer's combine table is the exact transpose of its dispatch table;
``step-hop-bound``
    ``bytes <= total_new_tokens * top_k * num_layers * 2 * vector_bytes``.

The last one is the identity that makes source replication visible. Replicating
a table over a ``W``-rank group multiplies its bytes by ``W``, and a wide EP
world leaves no slack for that factor: at ``W = 8`` a token reaches at most 7
remote owners against a ``top_k`` of 8, so a replicated table exceeds the bound
several times over. At ``W = 2`` a token reaches at most 1 remote owner, the
correct table uses an eighth of the bound, and the same replication fits
inside it undetected. The bound has to be evaluated at a wide EP world to mean
anything.

Four further rules need deduplicated captured routing and are applied only to
it:

``vector-granularity``
    every directed byte count is a positive multiple of one hidden vector;
``request-identity``
    every attributed request is a scheduled request with new tokens;
``per-request-hop-bound`` and ``per-layer-hop-bound``
    per layer and phase, bytes are at most
    ``tokens * min(top_k, W - 1) * vector_bytes``.

They are excluded from the uniform approximation on purpose: that
approximation spreads ``total_new_tokens * top_k`` assignments evenly over the
group without merging several selected experts that land on the same
destination, so it exceeds ``min(top_k, W - 1)`` per token whenever
``top_k > W`` by construction rather than by defect.

A violated rule is fatal and unscored. It voids the run that produced the
table; it is never reported as a fraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: routed byte evidence derived from deduplicated captured routing
ROUTED_EVIDENCE_CAPTURED = "captured-routing"
#: routed byte evidence derived from the uniform destination approximation
ROUTED_EVIDENCE_UNIFORM = "uniform-approximation"
ROUTED_EVIDENCE_MODES = (ROUTED_EVIDENCE_CAPTURED, ROUTED_EVIDENCE_UNIFORM)

#: rules that hold for every routed representation
ALWAYS_APPLIED_RULES = (
    "source-attribution",
    "destination-legality",
    "owner-egress",
    "transpose-symmetry",
    "step-hop-bound",
)
#: rules that need deduplicated captured routing
CAPTURED_ONLY_RULES = (
    "vector-granularity",
    "request-identity",
    "per-request-hop-bound",
    "per-layer-hop-bound",
)
CONSERVATION_RULES = ALWAYS_APPLIED_RULES + CAPTURED_ONLY_RULES

#: the two all-to-allv phases of one MoE layer, in execution order
_PHASES = ("dispatch", "combine")


@dataclass(frozen=True)
class RoutedPhaseTable:
    """One layer-phase directed byte table, the object under test."""

    layer: int
    phase: str
    pair_payload_bytes: tuple[tuple[int, int, int], ...]
    request_pair_payload_bytes: tuple[tuple[str, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("table.phase: expected dispatch or combine")
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("table.layer: expected a nonnegative integer")


@dataclass(frozen=True)
class RoutedTokenOwnership:
    """Source-observed token ownership carried across the adapter seam.

    ``request_token_counts`` is the adapter's own per-request scheduled token
    count, in the record's request order, and ``engine_rank`` is the one rank
    declared to dispatch them. Peer expert-parallel ranks own experts and carry
    no scheduled tokens.
    """

    engine_rank: int
    request_token_counts: tuple[tuple[str, int], ...]
    num_layers: int
    top_k: int
    vector_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("engine_rank", self.engine_rank),
            ("num_layers", self.num_layers),
            ("top_k", self.top_k),
            ("vector_bytes", self.vector_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ownership.{name}: expected a nonnegative integer")
        if not isinstance(self.request_token_counts, tuple):
            raise TypeError("ownership.request_token_counts: expected a tuple")
        seen: set[str] = set()
        for index, entry in enumerate(self.request_token_counts):
            path = f"ownership.request_token_counts[{index}]"
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"{path}: expected a two-item tuple")
            request_id, tokens = entry
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError(f"{path}[0]: expected a nonblank request id")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise ValueError(f"{path}[1]: expected a positive token count")
            if request_id in seen:
                raise ValueError(f"{path}[0]: duplicate request identity")
            seen.add(request_id)

    @property
    def total_new_tokens(self) -> int:
        """The step's owned token count, summed over its requests."""

        return sum(tokens for _, tokens in self.request_token_counts)

    @property
    def owned_request_ids(self) -> frozenset[str]:
        """The request identities this owner dispatched tokens for."""

        return frozenset(request_id for request_id, _ in self.request_token_counts)

    @property
    def step_hop_bound(self) -> int:
        """``total_new_tokens * top_k * num_layers * 2`` hidden-vector hops."""

        return self.total_new_tokens * self.top_k * self.num_layers * 2

    @property
    def step_byte_bound(self) -> int:
        """The step hop bound expressed in directed bytes."""

        return self.step_hop_bound * self.vector_bytes

    def tokens_for(self, request_id: str) -> int:
        """The owned token count of one request, zero when it owns none."""

        for owned_id, tokens in self.request_token_counts:
            if owned_id == request_id:
                return tokens
        return 0


@dataclass(frozen=True)
class RoutedConservationReport:
    """What the guard observed, plus every rule it found violated."""

    ep_world: int
    owner_rank: int
    evidence_mode: str
    source_ranks: tuple[int, ...]
    total_directed_bytes: int
    owner_egress_bytes: int
    owner_ingress_bytes: int
    emitted_hops: int
    step_hop_bound: int
    per_layer_hop_bound: int
    per_layer_hops: tuple[tuple[int, str, int], ...]
    checked_rules: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def conserved(self) -> bool:
        """Whether every checked rule held."""

        return not self.violations

    def require_conserved(self) -> RoutedConservationReport:
        """Return the report, raising when any checked rule was violated."""

        if self.violations:
            raise ValueError(
                "routed MoE byte conservation failed at EP world "
                f"{self.ep_world}: " + ", ".join(self.violations)
            )
        return self


def _transpose(rows: Sequence[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((destination, source, size) for source, destination, size in rows))


def _transpose_requests(
    rows: Sequence[tuple[str, int, int, int]],
) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        sorted(
            (request_id, destination, source, size)
            for request_id, source, destination, size in rows
        )
    )


def routed_moe_conservation_report(
    tables: Sequence[RoutedPhaseTable],
    ownership: RoutedTokenOwnership,
    ep_ranks: Sequence[int],
    *,
    evidence_mode: str,
) -> RoutedConservationReport:
    """Check one step's routed byte tables against independent ownership.

    ``tables`` is the projection under test. Every other argument comes from
    the step record, the model geometry and the declared expert group, so the
    check never consumes the routing walk that produced ``tables``.
    """

    if not isinstance(ownership, RoutedTokenOwnership):
        raise TypeError("ownership must be a RoutedTokenOwnership")
    if evidence_mode not in ROUTED_EVIDENCE_MODES:
        known = ", ".join(ROUTED_EVIDENCE_MODES)
        raise ValueError(f"evidence_mode must be one of {known}; got {evidence_mode!r}")
    ranks = tuple(ep_ranks)
    if len(ranks) != len(set(ranks)):
        raise ValueError("ep_ranks: contains duplicate ranks")
    if ownership.engine_rank not in ranks:
        raise ValueError("ownership.engine_rank: outside the expert-parallel group")
    if ownership.vector_bytes <= 0:
        raise ValueError("ownership.vector_bytes: expected a positive integer")
    table_rows = tuple(tables)
    for table in table_rows:
        if not isinstance(table, RoutedPhaseTable):
            raise TypeError("tables must contain RoutedPhaseTable entries")

    captured = evidence_mode == ROUTED_EVIDENCE_CAPTURED
    checked = ALWAYS_APPLIED_RULES + (CAPTURED_ONLY_RULES if captured else ())
    violations: list[str] = []

    def flag(rule: str) -> None:
        if rule not in violations:
            violations.append(rule)

    world = len(ranks)
    legal_ranks = set(ranks)
    owner = ownership.engine_rank
    per_token_ceiling = min(ownership.top_k, max(world - 1, 0))
    per_layer_hop_bound = ownership.total_new_tokens * per_token_ceiling
    per_layer_byte_bound = per_layer_hop_bound * ownership.vector_bytes

    sources: set[int] = set()
    total_bytes = 0
    owner_egress = 0
    owner_ingress = 0
    owner_endpoint_bytes = 0
    per_layer_hops: list[tuple[int, str, int]] = []
    by_layer: dict[int, dict[str, RoutedPhaseTable]] = {}

    for table in table_rows:
        by_layer.setdefault(table.layer, {})[table.phase] = table
        phase_bytes = 0
        dispatching = table.phase == "dispatch"
        for source, destination, size in table.pair_payload_bytes:
            sources.add(source)
            total_bytes += size
            phase_bytes += size
            if source == owner:
                owner_egress += size
            if destination == owner:
                owner_ingress += size
            if (source == owner) != (destination == owner):
                owner_endpoint_bytes += size
            if dispatching and source != owner:
                flag("source-attribution")
            if not dispatching and destination != owner:
                flag("source-attribution")
            if destination not in legal_ranks or destination == source:
                flag("destination-legality")
            if source not in legal_ranks:
                flag("destination-legality")
            if captured and (size <= 0 or size % ownership.vector_bytes):
                flag("vector-granularity")
        per_layer_hops.append(
            (table.layer, table.phase, phase_bytes // ownership.vector_bytes)
        )
        if captured and phase_bytes > per_layer_byte_bound:
            flag("per-layer-hop-bound")

        if captured:
            per_request: dict[str, int] = {}
            for request_id, _, _, size in table.request_pair_payload_bytes:
                if request_id not in ownership.owned_request_ids:
                    flag("request-identity")
                per_request[request_id] = per_request.get(request_id, 0) + size
            for request_id, size in per_request.items():
                ceiling = (
                    ownership.tokens_for(request_id)
                    * per_token_ceiling
                    * ownership.vector_bytes
                )
                if size > ceiling:
                    flag("per-request-hop-bound")

    if owner_endpoint_bytes != total_bytes or owner_egress != owner_ingress:
        flag("owner-egress")
    if total_bytes > ownership.step_byte_bound:
        flag("step-hop-bound")

    for phases in by_layer.values():
        dispatch = phases.get("dispatch")
        combine = phases.get("combine")
        if dispatch is None or combine is None:
            flag("transpose-symmetry")
            continue
        if tuple(sorted(combine.pair_payload_bytes)) != _transpose(
            dispatch.pair_payload_bytes
        ):
            flag("transpose-symmetry")
            continue
        has_requests = bool(
            dispatch.request_pair_payload_bytes or combine.request_pair_payload_bytes
        )
        if has_requests and tuple(
            sorted(combine.request_pair_payload_bytes)
        ) != _transpose_requests(dispatch.request_pair_payload_bytes):
            flag("transpose-symmetry")

    return RoutedConservationReport(
        ep_world=world,
        owner_rank=owner,
        evidence_mode=evidence_mode,
        source_ranks=tuple(sorted(sources)),
        total_directed_bytes=total_bytes,
        owner_egress_bytes=owner_egress,
        owner_ingress_bytes=owner_ingress,
        emitted_hops=total_bytes // ownership.vector_bytes,
        step_hop_bound=ownership.step_hop_bound,
        per_layer_hop_bound=per_layer_hop_bound,
        per_layer_hops=tuple(per_layer_hops),
        checked_rules=checked,
        violations=tuple(violations),
    )


__all__ = [
    "ALWAYS_APPLIED_RULES",
    "CAPTURED_ONLY_RULES",
    "CONSERVATION_RULES",
    "ROUTED_EVIDENCE_CAPTURED",
    "ROUTED_EVIDENCE_MODES",
    "ROUTED_EVIDENCE_UNIFORM",
    "RoutedConservationReport",
    "RoutedPhaseTable",
    "RoutedTokenOwnership",
    "routed_moe_conservation_report",
]
