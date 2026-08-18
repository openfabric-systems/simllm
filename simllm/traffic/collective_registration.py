"""One-time NCCL/RCCL channel-and-buffer registration as an explicit time cost.

The maintainer's 2026-08-18 kernel-time determinism ruling makes collective
work the single exception to the deterministic-constant contract. Until the
packetized path over the GPU's NVLink, xGMI or UALink ports lands, a collective
completes through the deterministic ATLAHS/htsim chain with a no-tail constant
completion, *gated on the destination memory being registered*, and that
registration carries an explicit modeled time cost. This module owns that cost.

Semantics. Before the first collective can use a
``(communicator, generation, channel, buffer)`` identity, the registration cost
is paid once and serialized ahead of that collective's own completion. Every
later collective on a registered identity pays nothing. Three events force a
re-registration and are the only three:

- **new buffer**: a destination buffer whose site this communicator and channel
  have not registered;
- **new peer**: a different participant rank set, which is a different
  communicator by construction;
- **communicator rebuild**: an explicit destroy-and-init cycle, modeled as a
  generation bump that invalidates every channel and buffer of that
  communicator at once.

Provenance, stated narrowly. Exactly two things here come from evidence. The
net plugin ABI recorded in ``docs/papers/amd-gpu-fabric.md`` establishes that a
registration entry point *exists* (``regMr`` and ``regMrDmaBuf`` are members of
the documented ``ncclNet_v6`` struct, and the audited ``net_v12.h`` of NCCL
``v2.30.7-1`` carries the same ``regMr`` member) and that the *same ABI serves
both stacks*, since RCCL exposes it under ``librccl-net.so``. Everything else
below is a declared model choice, not a transcription of the ABI: that the cost
is paid once rather than per call, that the identity is scoped to a buffer,
that a channel is part of that identity, that exactly three events force a
re-registration, and of course the 20 microsecond duration. Asking a declared
cost for a calibrated value fails closed rather than returning the declared
number under a calibrated label; TRAF-56 is the calibration task, and it has to
measure the model choices as well as the constant.

Layering, and where it is not yet clean. This module owns the cost model, the
identity rules and the ledger, and :mod:`simllm.backends.step_sink` is where
the charge reaches the live metric chain; those two are one authority with an
explicit projection. The mirrored stack seam in
:mod:`simllm.compute.nccl_stack` is *not* joined to them: its
``require_buffer_registration`` gate keeps its own per-communicator
``registered_buffers`` state, that state carries no generation, and the live
chain never consults it. So a run today has two registration states that agree
only by convention. TRAF-58 unifies them into one gate and one authority.

Default off. A configuration that does not name a registration model builds a
disabled ledger that charges zero for every identity, so every accepted
artifact, timestamp and metric stays byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from simllm.core.execution import CollectiveWork

#: how far a registration cost may be trusted
#:
#: ``declared`` is a configuration constant that no measurement anchors.
#: ``calibrated`` is fitted to a capture of the very thing it prices and must
#: name that capture. There is deliberately no middle class: a registration
#: cost is either measured or declared, and the point of the distinction is
#: that a consumer asking for the first must never be served the second.
COLLECTIVE_REGISTRATION_EVIDENCE_CLASSES = ("declared", "calibrated")

#: the declared one-time cost of registering one channel-buffer identity, ps
#:
#: Twenty microseconds. Nothing in this repository measures it. It is the order
#: of magnitude implied by RDMA memory registration pinning pages at
#: communicator setup, and it is a placeholder until TRAF-56 measures one.
DECLARED_CHANNEL_REGISTRATION_COST_PS = 20_000_000

#: the one buffer-identity policy this module implements
SEMANTIC_SITE_BUFFER_POLICY = "semantic-site"
COLLECTIVE_BUFFER_POLICIES = (SEMANTIC_SITE_BUFFER_POLICY,)

_STEP_PREFIX = re.compile(r"^step-\d+:")


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


@dataclass(frozen=True)
class CollectiveRegistrationProvenance:
    """Where one registration cost came from and how far it is trusted.

    ``source`` and ``locator`` identify the evidence, ``basis`` states in one
    sentence what that evidence does and does not establish, and
    ``measurement`` names the capture a calibrated cost was fitted to. A
    ``calibrated`` record without a measurement cannot be constructed, and a
    ``declared`` record with one is equally refused: the two classes are not
    interchangeable spellings.
    """

    evidence_class: str
    source: str
    locator: str
    basis: str
    measurement: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_class not in COLLECTIVE_REGISTRATION_EVIDENCE_CLASSES:
            raise ValueError(
                "evidence_class must be one of "
                f"{COLLECTIVE_REGISTRATION_EVIDENCE_CLASSES}, got "
                f"{self.evidence_class!r}"
            )
        for name in ("source", "locator", "basis"):
            _require_text(name, getattr(self, name))
        if self.evidence_class == "calibrated":
            if self.measurement is None:
                raise ValueError(
                    "a calibrated registration cost must name the measurement it "
                    "was fitted to"
                )
            _require_text("measurement", self.measurement)
        elif self.measurement is not None:
            raise ValueError(
                "a declared registration cost cannot name a measurement; declare "
                "it calibrated or drop the measurement"
            )


@dataclass(frozen=True)
class CollectiveRegistrationCost:
    """One immutable per-identity registration cost with its provenance."""

    cost_id: str
    registration_cost_ps: int
    provenance: CollectiveRegistrationProvenance

    def __post_init__(self) -> None:
        _require_text("cost_id", self.cost_id)
        _require_int("registration_cost_ps", self.registration_cost_ps, minimum=0)
        if not isinstance(self.provenance, CollectiveRegistrationProvenance):
            raise TypeError("provenance must be a CollectiveRegistrationProvenance")

    @property
    def evidence_class(self) -> str:
        """Return the declared evidence class of this cost."""

        return self.provenance.evidence_class

    def declared_cost_ps(self) -> int:
        """Return the cost as configuration, whatever its evidence class."""

        return self.registration_cost_ps

    def calibrated_cost_ps(self) -> int:
        """Return the cost, refusing a value that no measurement anchors.

        This is the fail-closed hook. A caller that needs a measured
        registration cost calls this and is refused until TRAF-56 lands one,
        rather than silently receiving the declared constant.
        """

        if self.provenance.evidence_class != "calibrated":
            raise ValueError(
                f"registration cost {self.cost_id!r} is "
                f"{self.provenance.evidence_class!r}, not calibrated: no "
                "measurement anchors it, so no calibrated value exists. Land "
                "TRAF-56 or ask for declared_cost_ps and publish the class"
            )
        return self.registration_cost_ps


DECLARED_NCCL_CHANNEL_REGISTRATION_COST = CollectiveRegistrationCost(
    cost_id="nccl-channel-registration-declared-v1",
    registration_cost_ps=DECLARED_CHANNEL_REGISTRATION_COST_PS,
    provenance=CollectiveRegistrationProvenance(
        evidence_class="declared",
        source=(
            "the NCCL and RCCL net plugin ABI as recorded in "
            "docs/papers/amd-gpu-fabric.md, together with the BACK-47 "
            "registration of that plugin seam"
        ),
        locator=(
            "the documented ncclNet_v6 regMr and regMrDmaBuf members, which NCCL "
            "calls so an RDMA NIC can prepare a buffer and which RCCL exposes "
            "unchanged under librccl-net.so, together with the regMr member of "
            "src/include/plugin/net/net_v12.h in the NCCL v2.30.7-1 release this "
            "repository's name audit pins"
        ),
        basis=(
            "the ABI establishes two things and no more: that a registration "
            "entry point exists at the plugin seam, and that one seam serves "
            "both NCCL and RCCL. That the cost is paid once rather than per "
            "call, that the identity is scoped to a buffer, that a channel "
            "belongs to that identity, that exactly three events force a "
            "re-registration, and the 20 microsecond duration itself are all "
            "declared model choices with no measurement behind them"
        ),
    ),
)


@dataclass(frozen=True)
class CollectiveRegistrationIdentity:
    """The one thing a registration registers.

    Two collectives share a registration exactly when all four fields agree.
    ``generation`` is what a communicator rebuild moves; the other three are
    what a new peer set, a new channel and a new buffer move.
    """

    communicator_id: str
    generation: int
    channel_id: int
    buffer_id: str

    def __post_init__(self) -> None:
        _require_text("communicator_id", self.communicator_id)
        _require_int("generation", self.generation, minimum=0)
        _require_int("channel_id", self.channel_id, minimum=0)
        _require_text("buffer_id", self.buffer_id)

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form of this identity."""

        return {
            "communicator_id": self.communicator_id,
            "generation": self.generation,
            "channel_id": self.channel_id,
            "buffer_id": self.buffer_id,
        }


@dataclass(frozen=True)
class CollectiveRegistrationModel:
    """A named, immutable registration policy: what costs what, per channel."""

    model_id: str
    cost: CollectiveRegistrationCost
    channels_per_communicator: int = 1
    buffer_policy: str = SEMANTIC_SITE_BUFFER_POLICY

    def __post_init__(self) -> None:
        _require_text("model_id", self.model_id)
        if not isinstance(self.cost, CollectiveRegistrationCost):
            raise TypeError("cost must be a CollectiveRegistrationCost")
        _require_int(
            "channels_per_communicator",
            self.channels_per_communicator,
            minimum=1,
        )
        if self.buffer_policy not in COLLECTIVE_BUFFER_POLICIES:
            raise ValueError(
                f"buffer_policy must be one of {COLLECTIVE_BUFFER_POLICIES}, got "
                f"{self.buffer_policy!r}"
            )

    @property
    def channel_ids(self) -> tuple[int, ...]:
        """Return the channels one communicator registers separately."""

        return tuple(range(self.channels_per_communicator))

    def first_use_cost_ps(self) -> int:
        """Return what one communicator-buffer pair costs across its channels."""

        return self.channels_per_communicator * self.cost.registration_cost_ps


NCCL_CHANNEL_REGISTRATION_MODEL = CollectiveRegistrationModel(
    model_id="nccl-channel-registration-v1",
    cost=DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
)

_NAMED_COLLECTIVE_REGISTRATION_MODELS = (NCCL_CHANNEL_REGISTRATION_MODEL,)


@dataclass(frozen=True)
class CollectiveRegistrationEvent:
    """One charged registration, retained as the ledger's read-only record."""

    identity: CollectiveRegistrationIdentity
    cost_ps: int
    reason: str
    operation_id: str | None = None
    step_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CollectiveRegistrationIdentity):
            raise TypeError("identity must be a CollectiveRegistrationIdentity")
        _require_int("cost_ps", self.cost_ps, minimum=0)
        _require_text("reason", self.reason)
        if self.operation_id is not None:
            _require_text("operation_id", self.operation_id)
        if self.step_index is not None:
            _require_int("step_index", self.step_index, minimum=0)

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form of this event."""

        return {
            "identity": self.identity.to_json(),
            "cost_ps": self.cost_ps,
            "reason": self.reason,
            "operation_id": self.operation_id,
            "step_index": self.step_index,
        }


#: the three reasons a charge happens, and the one reason it does not
FIRST_USE_REASON = "first-use"
NEW_BUFFER_REASON = "new-buffer"
NEW_PEER_REASON = "new-peer"
REBUILD_REASON = "communicator-rebuild"
COLLECTIVE_REGISTRATION_REASONS = (
    FIRST_USE_REASON,
    NEW_BUFFER_REASON,
    NEW_PEER_REASON,
    REBUILD_REASON,
)


def collective_communicator_id(work: CollectiveWork) -> str:
    """Return the communicator identity of one semantic collective.

    A communicator is its participant rank set together with the collective
    kind that group serves, which is how a framework builds one. Changing the
    rank set is therefore a new communicator, and that is exactly the new-peer
    re-registration event.
    """

    if not isinstance(work, CollectiveWork):
        raise TypeError("work must be a CollectiveWork")
    if len(work.ranks) < 2:
        raise ValueError("a collective communicator needs at least two ranks")
    ranks = "-".join(str(rank) for rank in work.ranks)
    return f"{work.collective}:[{ranks}]"


def collective_buffer_id(
    operation_id: str,
    *,
    buffer_policy: str = SEMANTIC_SITE_BUFFER_POLICY,
) -> str:
    """Return the destination-buffer identity of one collective operation.

    The ``semantic-site`` policy strips the leading ``step-<n>:`` component, so
    the buffer is the collective's site rather than its occurrence. A
    deployment that runs the same layer's all-reduce every step registers that
    buffer once and never again, which is what a real framework does with a
    reused communication buffer.

    An operation identity without that component fails closed. Charging every
    step because the producer names its operations differently would be a
    silent modeling error, and guessing which of two spellings means the same
    buffer is not this module's decision to make.
    """

    _require_text("operation_id", operation_id)
    if buffer_policy not in COLLECTIVE_BUFFER_POLICIES:
        raise ValueError(
            f"buffer_policy must be one of {COLLECTIVE_BUFFER_POLICIES}, got "
            f"{buffer_policy!r}"
        )
    stripped = _STEP_PREFIX.sub("", operation_id, count=1)
    if stripped == operation_id:
        raise ValueError(
            f"operation identity {operation_id!r} carries no 'step-<n>:' "
            "component, so the semantic-site buffer policy cannot tell a reused "
            "buffer from a new one. Name the operation with the repository's "
            "step-scoped identity or register a buffer policy for this producer"
        )
    if not stripped.strip():
        raise ValueError(
            f"operation identity {operation_id!r} is only a step component"
        )
    return stripped


@dataclass
class CollectiveRegistrationLedger:
    """The one mutable registration authority for a deployment.

    A ledger built with ``model=None`` is disabled: it charges zero for every
    identity, records no event and is the exact off path. A ledger built with a
    model charges the model's cost once per identity, in call order, and keeps
    a read-only event list as the evidence a run record publishes.
    """

    model: CollectiveRegistrationModel | None = None
    _generations: dict[str, int] = field(default_factory=dict, repr=False)
    _registered: set[CollectiveRegistrationIdentity] = field(
        default_factory=set,
        repr=False,
    )
    _events: list[CollectiveRegistrationEvent] = field(
        default_factory=list,
        repr=False,
    )
    _rebuilt: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.model is not None and not isinstance(
            self.model, CollectiveRegistrationModel
        ):
            raise TypeError("model must be a CollectiveRegistrationModel or None")

    @property
    def enabled(self) -> bool:
        """Whether this ledger charges anything at all."""

        return self.model is not None

    @property
    def events(self) -> tuple[CollectiveRegistrationEvent, ...]:
        """Return every charged registration, in charge order."""

        return tuple(self._events)

    @property
    def registered_identities(self) -> tuple[CollectiveRegistrationIdentity, ...]:
        """Return the registered identities, ordered by first charge."""

        return tuple(event.identity for event in self._events)

    @property
    def charged_ps(self) -> int:
        """Return the total registration time this ledger has charged."""

        return sum(event.cost_ps for event in self._events)

    def generation(self, communicator_id: str) -> int:
        """Return the current generation of one communicator."""

        _require_text("communicator_id", communicator_id)
        return self._generations.get(communicator_id, 0)

    def rebuild_communicator(self, communicator_id: str) -> int:
        """Model one destroy-and-init cycle; return the new generation.

        Every channel and buffer of that communicator re-registers on next use.
        A rebuild of a communicator the ledger has never seen is legal and
        simply starts it one generation later, because a caller cannot be
        required to know whether a collective has run yet.
        """

        _require_text("communicator_id", communicator_id)
        generation = self._generations.get(communicator_id, 0) + 1
        self._generations[communicator_id] = generation
        self._rebuilt.add(communicator_id)
        return generation

    def rebuild_all(self) -> None:
        """Rebuild every communicator this ledger has charged."""

        for communicator_id in sorted(
            {identity.communicator_id for identity in self._registered}
        ):
            self.rebuild_communicator(communicator_id)

    def identities_for(
        self,
        *,
        communicator_id: str,
        buffer_id: str,
    ) -> tuple[CollectiveRegistrationIdentity, ...]:
        """Return the identities one communicator-buffer pair needs."""

        if self.model is None:
            return ()
        _require_text("communicator_id", communicator_id)
        _require_text("buffer_id", buffer_id)
        generation = self.generation(communicator_id)
        return tuple(
            CollectiveRegistrationIdentity(
                communicator_id=communicator_id,
                generation=generation,
                channel_id=channel_id,
                buffer_id=buffer_id,
            )
            for channel_id in self.model.channel_ids
        )

    def is_registered(self, identity: CollectiveRegistrationIdentity) -> bool:
        """Whether one identity has already paid."""

        if not isinstance(identity, CollectiveRegistrationIdentity):
            raise TypeError("identity must be a CollectiveRegistrationIdentity")
        return identity in self._registered

    def charge(
        self,
        *,
        communicator_id: str,
        buffer_id: str,
        operation_id: str | None = None,
        step_index: int | None = None,
    ) -> int:
        """Charge every unregistered channel of one communicator-buffer pair.

        Returns the picoseconds this call owes, which is zero on every use
        after the first. The cost is serialized ahead of the collective that
        triggered it; this ledger only decides the amount.
        """

        if self.model is None:
            return 0
        charged_ps = 0
        for identity in self.identities_for(
            communicator_id=communicator_id,
            buffer_id=buffer_id,
        ):
            if identity in self._registered:
                continue
            cost_ps = self.model.cost.registration_cost_ps
            reason = self._reason_for(identity)
            self._registered.add(identity)
            self._events.append(
                CollectiveRegistrationEvent(
                    identity=identity,
                    cost_ps=cost_ps,
                    reason=reason,
                    operation_id=operation_id,
                    step_index=step_index,
                )
            )
            charged_ps += cost_ps
        return charged_ps

    def charge_collective(
        self,
        work: CollectiveWork,
        operation_id: str,
        *,
        step_index: int | None = None,
    ) -> int:
        """Charge the registration one semantic collective needs at first use."""

        if self.model is None:
            return 0
        return self.charge(
            communicator_id=collective_communicator_id(work),
            buffer_id=collective_buffer_id(
                operation_id,
                buffer_policy=self.model.buffer_policy,
            ),
            operation_id=operation_id,
            step_index=step_index,
        )

    def _reason_for(self, identity: CollectiveRegistrationIdentity) -> str:
        """Classify one about-to-be-charged identity. Called before it is added.

        The classification is evidence for a reader, never an input to the
        amount: every charge costs the same. It answers which of the three
        re-registration events the reader is looking at, or first use.
        """

        if identity.generation > 0 and identity.communicator_id in self._rebuilt:
            #: Once rebuilt, a communicator keeps this label for every later
            #: charge, including a genuinely new buffer at the new generation.
            #: The label is evidence for a reader and never an input to the
            #: amount, so the coarseness costs no picosecond; sharpening it
            #: would need a per-generation buffer history.
            return REBUILD_REASON
        same_communicator = {
            registered
            for registered in self._registered
            if registered.communicator_id == identity.communicator_id
            and registered.generation == identity.generation
        }
        if same_communicator:
            if any(
                registered.buffer_id == identity.buffer_id
                for registered in same_communicator
            ):
                #: another channel of a buffer already being registered now
                return FIRST_USE_REASON
            return NEW_BUFFER_REASON
        kind = identity.communicator_id.split(":", 1)[0]
        if any(
            registered.communicator_id.split(":", 1)[0] == kind
            for registered in self._registered
        ):
            return NEW_PEER_REASON
        return FIRST_USE_REASON


def resolve_collective_registration(
    selector: str | CollectiveRegistrationModel | None,
) -> CollectiveRegistrationModel | None:
    """Resolve a named registration model, or ``None`` for the exact off path."""

    if selector is None:
        return None
    if isinstance(selector, CollectiveRegistrationModel):
        return selector
    if isinstance(selector, str):
        for model in _NAMED_COLLECTIVE_REGISTRATION_MODELS:
            if selector == model.model_id:
                return model
        raise ValueError(f"unknown collective registration model {selector!r}")
    raise TypeError(
        "collective registration must be None, a selector string, or a "
        "CollectiveRegistrationModel"
    )


__all__ = [
    "COLLECTIVE_BUFFER_POLICIES",
    "COLLECTIVE_REGISTRATION_EVIDENCE_CLASSES",
    "COLLECTIVE_REGISTRATION_REASONS",
    "DECLARED_CHANNEL_REGISTRATION_COST_PS",
    "DECLARED_NCCL_CHANNEL_REGISTRATION_COST",
    "FIRST_USE_REASON",
    "NCCL_CHANNEL_REGISTRATION_MODEL",
    "NEW_BUFFER_REASON",
    "NEW_PEER_REASON",
    "REBUILD_REASON",
    "SEMANTIC_SITE_BUFFER_POLICY",
    "CollectiveRegistrationCost",
    "CollectiveRegistrationEvent",
    "CollectiveRegistrationIdentity",
    "CollectiveRegistrationLedger",
    "CollectiveRegistrationModel",
    "CollectiveRegistrationProvenance",
    "collective_buffer_id",
    "collective_communicator_id",
    "resolve_collective_registration",
]
