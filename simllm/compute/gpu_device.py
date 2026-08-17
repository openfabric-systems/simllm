"""GPU device composition with typed PCIe and NVLink ports (COMP-34).

The NIC has been a device with typed ports since BACK-18: ``RnicDevice`` is
composed from a work-queue core plus optional modules, and a disabled module
keeps its interface with its parameters inert or explicitly rejected. The GPU
was a calibration profile carrying two separate link mechanisms with no port
object over them. This module adds that object, and only that object.

The mechanisms stay authoritative and unchanged:

- the host link is priced by the per-direction copy-engine profiles
  (:class:`~simllm.compute.gpu_model.CopyDirectionProfile` inside
  :class:`~simllm.compute.gpu_model.CopyEngineProfile`, serviced by
  :class:`~simllm.compute.gpu_model.CopyEngineServiceModel`),
- the peer link is priced by the flat per-GPU NVLink egress cursor
  (:class:`~simllm.compute.gpu_model.NvlinkProfile`) for SM stores and by the
  peer copy direction for copy-engine peer transfers.

A port carries protocol identity, direction, ceiling, declared capabilities and
the provenance of that ceiling, behind one versioned configuration. It is not a
second timing mechanism, and it never introduces a rate of its own: with no
declared ceiling, :attr:`GpuDevice.architecture` is the input architecture
object itself, so every accepted timestamp, counter and byte count is
reproduced exactly.

Two semantics are load bearing and are easy to get wrong:

1. **A disabled port is a declaration that is absent, not a mechanism that is
   off.** Disabling a port never rescopes the copy engine or the egress cursor:
   the pre-existing mechanism stays exactly as it is and stays authoritative,
   the port keeps its interface and is still reported with a not-applicable
   applicability, its own parameters are inert, and any request made of the
   port is rejected with a diagnostic naming it.
2. **Reading a ceiling is not declaring one.** A port with no declared ceiling
   reads its effective ceiling out of the mechanism it wraps and carries
   ``CALIBRATION_DERIVED`` provenance. A port with a declared ceiling replaces
   the mechanism parameter for the directions that port carries, and only
   those, and the derived architecture is renamed so no artifact can claim the
   base profile identity while carrying a rescoped parameter.

Everything a port cannot do fails closed at configuration time rather than at
first use: a capability with no mechanism behind it, an xGMI port (COMP-35 owns
vendor instantiation), a peer-store port on a calibration with no NVLink
profile, a copy direction the engine does not declare, or two ports claiming
one mechanism.

Scope boundary. Peer topology, per-link routing, ingress service and reduction
lanes stay with COMP-31. Making the ABI v2 packet vocabulary reachable from a
non-wire port stays with BACK-48, so the transport-control capabilities are
declared here only so a request for one is rejected by name.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum

from simllm.compute.gpu_model import (
    CopyDirection,
    CopyEngineProfile,
    CopyEngineServiceModel,
    GpuArchitectureProfile,
    SmSchedulerModel,
)

GPU_DEVICE_IMPLEMENTATION = "simllm-gpu-device-v1"
GPU_DEVICE_CONFIG_VERSION = 1
GPU_PORT_CONFIG_VERSION = 1


class GpuPortProtocol(str, Enum):
    """Wire protocol a port speaks.

    ``XGMI`` is nameable so an AMD topology can be described, but a port that
    claims it is rejected: no first-party AMD measurement exists in this
    repository and COMP-35 owns vendor instantiation.
    """

    PCIE = "pcie"
    NVLINK_C2C = "nvlink_c2c"
    NVLINK = "nvlink"
    XGMI = "xgmi"


class GpuPortRole(str, Enum):
    """Which link of the port taxonomy a port sits on."""

    HOST_LINK = "host_link"
    PEER_LINK = "peer_link"


class GpuPortDirection(str, Enum):
    """Direction relative to the GPU that owns the port.

    ``INGRESS`` enters the GPU, ``EGRESS`` leaves it. A link that is measured
    asymmetric is expressed as two ports rather than one bidirectional port, so
    a single host rate can never be applied to the direction it does not fit.
    """

    INGRESS = "ingress"
    EGRESS = "egress"
    BIDIRECTIONAL = "bidirectional"


class GpuPortCapability(str, Enum):
    """What a port advertises.

    The first three have a mechanism behind them in this repository. The
    transport-control three do not: they exist so that a request for one is
    rejected by name instead of silently accepted, and BACK-48 owns making the
    ABI v2 vocabulary reachable from a non-wire port.
    """

    COPY_ENGINE_TRANSFER = "copy_engine_transfer"
    PEER_STORE_EGRESS = "peer_store_egress"
    CEILING_OVERRIDE = "ceiling_override"
    ECN_MARKING = "ecn_marking"
    PRIORITY_FLOW_CONTROL = "priority_flow_control"
    CONGESTION_NOTIFICATION = "congestion_notification"


class GpuPortCeilingProvenance(str, Enum):
    """Where a port's ceiling comes from.

    ``CALIBRATION_DERIVED`` is reserved for a ceiling read out of the mechanism
    the port wraps; a caller may not claim it for a value it declares.
    """

    CALIBRATION_DERIVED = "calibration_derived"
    FIRST_PARTY_MEASURED = "first_party_measured"
    VENDOR_NAMEPLATE = "vendor_nameplate"
    MODEL_CONFIGURATION = "model_configuration"


class GpuPortApplicability(str, Enum):
    """Whether a declared port is live, mirroring ``RnicStageApplicability``."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


#: capabilities that name a mechanism this repository actually prices
MECHANISM_CAPABILITIES = (
    GpuPortCapability.COPY_ENGINE_TRANSFER,
    GpuPortCapability.PEER_STORE_EGRESS,
)

#: capabilities with no mechanism behind them today; declaring one is rejected
TRANSPORT_CONTROL_CAPABILITIES = (
    GpuPortCapability.ECN_MARKING,
    GpuPortCapability.PRIORITY_FLOW_CONTROL,
    GpuPortCapability.CONGESTION_NOTIFICATION,
)

_ROLE_PROTOCOLS: dict[GpuPortRole, tuple[GpuPortProtocol, ...]] = {
    GpuPortRole.HOST_LINK: (GpuPortProtocol.PCIE, GpuPortProtocol.NVLINK_C2C),
    GpuPortRole.PEER_LINK: (
        GpuPortProtocol.NVLINK,
        GpuPortProtocol.PCIE,
        GpuPortProtocol.XGMI,
    ),
}

_COPY_DIRECTION_ROLE: dict[CopyDirection, GpuPortRole] = {
    CopyDirection.HOST_TO_DEVICE: GpuPortRole.HOST_LINK,
    CopyDirection.DEVICE_TO_HOST: GpuPortRole.HOST_LINK,
    CopyDirection.PEER_TO_PEER: GpuPortRole.PEER_LINK,
}

_COPY_DIRECTION_DIRECTIONS: dict[CopyDirection, tuple[GpuPortDirection, ...]] = {
    CopyDirection.HOST_TO_DEVICE: (
        GpuPortDirection.INGRESS,
        GpuPortDirection.BIDIRECTIONAL,
    ),
    CopyDirection.DEVICE_TO_HOST: (
        GpuPortDirection.EGRESS,
        GpuPortDirection.BIDIRECTIONAL,
    ),
    CopyDirection.PEER_TO_PEER: (
        GpuPortDirection.EGRESS,
        GpuPortDirection.BIDIRECTIONAL,
    ),
}


@dataclass(frozen=True, kw_only=True)
class GpuPortCeiling:
    """One port ceiling with the provenance of its value.

    The ceiling is carried in the mechanism's own unit, bytes per cycle of the
    mechanism's clock, so that a port that reads a mechanism parameter carries
    exactly that parameter and no rounded conversion of it.
    :attr:`bytes_per_second` is the derived physical reading.
    """

    bytes_per_cycle: float
    clock_hz: int
    provenance: GpuPortCeilingProvenance
    source: str
    reference: str = ""

    def __post_init__(self) -> None:
        _require_positive_number("bytes_per_cycle", self.bytes_per_cycle)
        _require_positive_int("clock_hz", self.clock_hz)
        if not isinstance(self.provenance, GpuPortCeilingProvenance):
            raise TypeError("provenance must be a GpuPortCeilingProvenance")
        _require_text("source", self.source)
        if not isinstance(self.reference, str):
            raise TypeError("reference must be a string")

    @property
    def bytes_per_second(self) -> float:
        return self.bytes_per_cycle * self.clock_hz


@dataclass(frozen=True, kw_only=True)
class GpuPortConfig:
    """One declared GPU port, versioned independently of the device config."""

    port_id: str
    protocol: GpuPortProtocol
    role: GpuPortRole
    direction: GpuPortDirection
    version: int = GPU_PORT_CONFIG_VERSION
    enabled: bool = True
    capabilities: tuple[GpuPortCapability, ...] = ()
    copy_engine_id: str | None = None
    copy_directions: tuple[CopyDirection, ...] = ()
    declared_ceiling: GpuPortCeiling | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "copy_directions", tuple(self.copy_directions))
        if self.version != GPU_PORT_CONFIG_VERSION:
            raise ValueError(
                f"port config version {self.version!r} is not supported; "
                f"this build implements version {GPU_PORT_CONFIG_VERSION}"
            )
        _require_text("port_id", self.port_id)
        _require_enum("protocol", self.protocol, GpuPortProtocol)
        _require_enum("role", self.role, GpuPortRole)
        _require_enum("direction", self.direction, GpuPortDirection)
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        for capability in self.capabilities:
            _require_enum("capability", capability, GpuPortCapability)
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError(f"port {self.port_id!r} declares a duplicate capability")
        for direction in self.copy_directions:
            _require_enum("copy direction", direction, CopyDirection)
        if len(set(self.copy_directions)) != len(self.copy_directions):
            raise ValueError(f"port {self.port_id!r} declares a duplicate copy direction")
        self._validate_protocol()
        self._validate_capabilities()
        self._validate_copy_directions()
        self._validate_declared_ceiling()

    def _validate_protocol(self) -> None:
        if self.protocol is GpuPortProtocol.XGMI:
            raise ValueError(
                f"port {self.port_id!r} declares the xGMI protocol, which has no "
                "first-party or declared profile in this repository; vendor port "
                "instantiation is COMP-35 and must supply a declared xGMI ceiling "
                "with its own provenance before an AMD cell may run"
            )
        if self.protocol not in _ROLE_PROTOCOLS[self.role]:
            allowed = ", ".join(item.value for item in _ROLE_PROTOCOLS[self.role])
            raise ValueError(
                f"port {self.port_id!r} declares protocol {self.protocol.value!r} on "
                f"role {self.role.value!r}, which carries only: {allowed}"
            )

    def _validate_capabilities(self) -> None:
        for capability in self.capabilities:
            if capability in TRANSPORT_CONTROL_CAPABILITIES:
                raise ValueError(
                    f"port {self.port_id!r} declares capability "
                    f"{capability.value!r}, which no GPU port can service today: "
                    "the ABI v2 packet and transport-control vocabulary is "
                    "reachable only through a wire port, and BACK-48 owns making "
                    "it reachable from a non-wire port"
                )
        if self.enabled and not any(
            capability in self.capabilities for capability in MECHANISM_CAPABILITIES
        ):
            declared = ", ".join(item.value for item in MECHANISM_CAPABILITIES)
            raise ValueError(
                f"enabled port {self.port_id!r} declares no mechanism capability; "
                f"one of these is required so the port is not silently inert: "
                f"{declared}"
            )
        if GpuPortCapability.PEER_STORE_EGRESS in self.capabilities and (
            self.role is not GpuPortRole.PEER_LINK
            or self.direction is not GpuPortDirection.EGRESS
        ):
            raise ValueError(
                f"port {self.port_id!r} claims the peer-store egress cursor, which "
                "belongs to an egress peer-link port"
            )

    def _validate_copy_directions(self) -> None:
        carries_copy = GpuPortCapability.COPY_ENGINE_TRANSFER in self.capabilities
        if carries_copy and not self.copy_directions:
            raise ValueError(
                f"port {self.port_id!r} claims copy-engine transfers but names no "
                "copy direction"
            )
        if carries_copy and self.copy_engine_id is None:
            raise ValueError(
                f"port {self.port_id!r} claims copy-engine transfers but names no "
                "copy engine"
            )
        if not carries_copy and (self.copy_directions or self.copy_engine_id is not None):
            raise ValueError(
                f"port {self.port_id!r} names a copy engine or direction without "
                "declaring the copy-engine transfer capability"
            )
        if self.copy_engine_id is not None:
            _require_text("copy_engine_id", self.copy_engine_id)
        for direction in self.copy_directions:
            if direction is CopyDirection.DEVICE_TO_DEVICE:
                raise ValueError(
                    f"port {self.port_id!r} names the device-to-device copy "
                    "direction, which stays inside one GPU and crosses no port"
                )
            if _COPY_DIRECTION_ROLE[direction] is not self.role:
                raise ValueError(
                    f"port {self.port_id!r} on role {self.role.value!r} names copy "
                    f"direction {direction.value!r}, which belongs to role "
                    f"{_COPY_DIRECTION_ROLE[direction].value!r}"
                )
            if self.direction not in _COPY_DIRECTION_DIRECTIONS[direction]:
                allowed = ", ".join(
                    item.value for item in _COPY_DIRECTION_DIRECTIONS[direction]
                )
                raise ValueError(
                    f"port {self.port_id!r} declares direction "
                    f"{self.direction.value!r} but names copy direction "
                    f"{direction.value!r}, which is carried by: {allowed}"
                )

    def _validate_declared_ceiling(self) -> None:
        if self.declared_ceiling is None:
            return
        if not isinstance(self.declared_ceiling, GpuPortCeiling):
            raise TypeError("declared_ceiling must be a GpuPortCeiling or None")
        if not self.enabled:
            raise ValueError(
                f"disabled port {self.port_id!r} declares a ceiling; a disabled "
                "port keeps its interface with its parameters inert and never "
                "rescopes the mechanism it would have wrapped"
            )
        if GpuPortCapability.CEILING_OVERRIDE not in self.capabilities:
            raise ValueError(
                f"port {self.port_id!r} declares a ceiling without advertising the "
                f"{GpuPortCapability.CEILING_OVERRIDE.value!r} capability"
            )
        if self.declared_ceiling.provenance is GpuPortCeilingProvenance.CALIBRATION_DERIVED:
            raise ValueError(
                f"port {self.port_id!r} declares a ceiling claiming "
                "calibration-derived provenance, which is reserved for a value "
                "read out of the mechanism the port wraps"
            )

    def advertises(self, capability: GpuPortCapability) -> bool:
        _require_enum("capability", capability, GpuPortCapability)
        return capability in self.capabilities


@dataclass(frozen=True, kw_only=True)
class GpuDeviceConfig:
    """One versioned GPU device composition entry point."""

    device_id: str
    version: int = GPU_DEVICE_CONFIG_VERSION
    ports: tuple[GpuPortConfig, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        if self.version != GPU_DEVICE_CONFIG_VERSION:
            raise ValueError(
                f"device config version {self.version!r} is not supported; "
                f"this build implements version {GPU_DEVICE_CONFIG_VERSION}"
            )
        _require_text("device_id", self.device_id)
        if any(not isinstance(port, GpuPortConfig) for port in self.ports):
            raise TypeError("ports must contain GpuPortConfig records")
        port_ids = [port.port_id for port in self.ports]
        if len(set(port_ids)) != len(port_ids):
            raise ValueError(f"device {self.device_id!r} declares a duplicate port ID")
        self._validate_mechanism_authority()

    def _validate_mechanism_authority(self) -> None:
        claimed_copy: dict[tuple[str, CopyDirection], str] = {}
        peer_store_owner: str | None = None
        for port in self.ports:
            if not port.enabled:
                continue
            for direction in port.copy_directions:
                assert port.copy_engine_id is not None
                key = (port.copy_engine_id, direction)
                owner = claimed_copy.get(key)
                if owner is not None:
                    raise ValueError(
                        f"ports {owner!r} and {port.port_id!r} both claim copy "
                        f"direction {direction.value!r} of engine "
                        f"{port.copy_engine_id!r}; one mechanism has one port "
                        "authority"
                    )
                claimed_copy[key] = port.port_id
            if port.advertises(GpuPortCapability.PEER_STORE_EGRESS):
                if peer_store_owner is not None:
                    raise ValueError(
                        f"ports {peer_store_owner!r} and {port.port_id!r} both claim "
                        "the peer-store egress cursor; the GPU has one flat egress "
                        "cursor and it has one port authority"
                    )
                peer_store_owner = port.port_id

    def port(self, port_id: str) -> GpuPortConfig:
        for port in self.ports:
            if port.port_id == port_id:
                return port
        raise KeyError(f"device {self.device_id!r} declares no port {port_id!r}")


@dataclass(frozen=True, kw_only=True)
class GpuPort:
    """One resolved port: its declaration plus the ceiling it actually carries.

    ``ceiling`` is ``None`` exactly when the port is disabled, because a
    disabled port's parameters are inert.
    """

    config: GpuPortConfig
    applicability: GpuPortApplicability
    ceiling: GpuPortCeiling | None

    @property
    def port_id(self) -> str:
        return self.config.port_id

    @property
    def protocol(self) -> GpuPortProtocol:
        return self.config.protocol

    @property
    def role(self) -> GpuPortRole:
        return self.config.role

    @property
    def direction(self) -> GpuPortDirection:
        return self.config.direction

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def capabilities(self) -> tuple[GpuPortCapability, ...]:
        return self.config.capabilities

    @property
    def copy_engine_id(self) -> str | None:
        return self.config.copy_engine_id

    @property
    def copy_directions(self) -> tuple[CopyDirection, ...]:
        return self.config.copy_directions

    def advertises(self, capability: GpuPortCapability) -> bool:
        return self.config.advertises(capability)

    def require(self, capability: GpuPortCapability) -> None:
        """Reject a capability this port does not advertise."""

        _require_enum("capability", capability, GpuPortCapability)
        if self.applicability is GpuPortApplicability.NOT_APPLICABLE:
            raise ValueError(
                f"port {self.port_id!r} is disabled, so capability "
                f"{capability.value!r} is not applicable; a disabled port keeps "
                "its interface and rejects every request made of it"
            )
        if not self.advertises(capability):
            advertised = ", ".join(item.value for item in self.capabilities) or "none"
            raise ValueError(
                f"port {self.port_id!r} does not advertise capability "
                f"{capability.value!r}; it advertises: {advertised}"
            )


@dataclass(frozen=True, kw_only=True)
class GpuPortReport:
    """Read-only projection of one resolved port, for studies and artifacts."""

    device_implementation: str
    device_id: str
    architecture_profile_id: str
    port_id: str
    protocol: str
    role: str
    direction: str
    enabled: bool
    applicability: str
    capabilities: tuple[str, ...]
    copy_engine_id: str
    copy_directions: tuple[str, ...]
    ceiling_bytes_per_cycle: float | None
    ceiling_bytes_per_second: float | None
    ceiling_clock_hz: int | None
    ceiling_provenance: str
    ceiling_source: str


class GpuDevice:
    """One composed GPU: an architecture profile plus its typed ports.

    Configuration-time rejection is the whole point of the constructor: a port
    that cannot be serviced, a capability with no mechanism, a mechanism with
    two port authorities and a disabled port asked to change a ceiling all fail
    here rather than at first use.
    """

    def __init__(
        self,
        config: GpuDeviceConfig,
        architecture: GpuArchitectureProfile,
    ) -> None:
        if not isinstance(config, GpuDeviceConfig):
            raise TypeError("config must be a GpuDeviceConfig")
        if not isinstance(architecture, GpuArchitectureProfile):
            raise TypeError("architecture must be a GpuArchitectureProfile")
        self.config = config
        self.base_architecture = architecture
        self._ports = tuple(self._resolve(port) for port in config.ports)
        self._ports_by_id = {port.port_id: port for port in self._ports}
        self.architecture = self._effective_architecture()

    @property
    def device_id(self) -> str:
        return self.config.device_id

    @property
    def ports(self) -> tuple[GpuPort, ...]:
        return self._ports

    def port(self, port_id: str) -> GpuPort:
        try:
            return self._ports_by_id[port_id]
        except KeyError:
            raise KeyError(
                f"device {self.device_id!r} declares no port {port_id!r}"
            ) from None

    def require_capability(
        self,
        port_id: str,
        capability: GpuPortCapability,
    ) -> GpuPort:
        """Return the port only if it advertises ``capability``."""

        port = self.port(port_id)
        port.require(capability)
        return port

    def sm_scheduler_model(self) -> SmSchedulerModel:
        """Return the kernel replay over this device's effective architecture."""

        return SmSchedulerModel(self.architecture)

    def copy_engine_service(self, engine_id: str) -> CopyEngineServiceModel:
        """Return the copy service over this device's effective architecture.

        A copy engine reachable only through a disabled port stays usable: the
        mechanism is authoritative and disabling a port removes a declaration,
        not a mechanism.
        """

        return CopyEngineServiceModel(self.architecture, engine_id)

    def port_reports(self) -> tuple[GpuPortReport, ...]:
        return tuple(
            GpuPortReport(
                device_implementation=GPU_DEVICE_IMPLEMENTATION,
                device_id=self.device_id,
                architecture_profile_id=self.architecture.profile_id,
                port_id=port.port_id,
                protocol=port.protocol.value,
                role=port.role.value,
                direction=port.direction.value,
                enabled=port.enabled,
                applicability=port.applicability.value,
                capabilities=tuple(item.value for item in port.capabilities),
                copy_engine_id=port.copy_engine_id or "",
                copy_directions=tuple(item.value for item in port.copy_directions),
                ceiling_bytes_per_cycle=(
                    None if port.ceiling is None else port.ceiling.bytes_per_cycle
                ),
                ceiling_bytes_per_second=(
                    None if port.ceiling is None else port.ceiling.bytes_per_second
                ),
                ceiling_clock_hz=(
                    None if port.ceiling is None else port.ceiling.clock_hz
                ),
                ceiling_provenance=(
                    "" if port.ceiling is None else port.ceiling.provenance.value
                ),
                ceiling_source=("" if port.ceiling is None else port.ceiling.source),
            )
            for port in self._ports
        )

    def _resolve(self, config: GpuPortConfig) -> GpuPort:
        # Mechanism binding is validated for a disabled port too: a declaration
        # that names an engine or a cursor the architecture does not have is a
        # configuration error whether or not the port is live, exactly as a
        # disabled RNIC module still hard-fails on a mismatched sub-config.
        mechanism = self._mechanism_ceiling(config)
        if not config.enabled:
            return GpuPort(
                config=config,
                applicability=GpuPortApplicability.NOT_APPLICABLE,
                ceiling=None,
            )
        declared = config.declared_ceiling
        if declared is None:
            return GpuPort(
                config=config,
                applicability=GpuPortApplicability.APPLICABLE,
                ceiling=mechanism,
            )
        assert mechanism is not None
        if declared.clock_hz != mechanism.clock_hz:
            raise ValueError(
                f"port {config.port_id!r} declares a ceiling on a "
                f"{declared.clock_hz} Hz clock while the mechanism it wraps runs "
                f"at {mechanism.clock_hz} Hz"
            )
        return GpuPort(
            config=config,
            applicability=GpuPortApplicability.APPLICABLE,
            ceiling=declared,
        )

    def _mechanism_ceiling(self, config: GpuPortConfig) -> GpuPortCeiling | None:
        """Return the ceiling the wrapped mechanism already carries.

        Rejects the port when the mechanism does not exist, when it does not
        declare a named direction, or when the directions one port carries
        disagree on their ceiling. The last case is the asymmetric host link:
        one port may not average two measured directions into one rate.
        """

        calibration = self.base_architecture.calibration
        values: list[tuple[str, float, int]] = []
        if config.advertises(GpuPortCapability.PEER_STORE_EGRESS):
            if calibration.nvlink is None:
                raise ValueError(
                    f"port {config.port_id!r} claims the peer-store egress cursor "
                    f"but calibration {calibration.calibration_id!r} declares no "
                    "NVLink profile; a port with no measured or declared profile "
                    "fails closed"
                )
            values.append(
                (
                    "nvlink egress cursor",
                    calibration.nvlink.bandwidth_bytes_per_cycle,
                    calibration.core_clock_hz,
                )
            )
        if config.advertises(GpuPortCapability.COPY_ENGINE_TRANSFER):
            engine = self._copy_engine(config)
            for direction in config.copy_directions:
                if direction not in engine.directions:
                    declared = ", ".join(item.value for item in engine.directions)
                    raise ValueError(
                        f"port {config.port_id!r} names copy direction "
                        f"{direction.value!r}, which copy engine "
                        f"{engine.engine_id!r} does not declare; it declares: "
                        f"{declared}"
                    )
                service = engine.service(direction)
                values.append(
                    (
                        f"copy engine {engine.engine_id} {direction.value}",
                        service.bandwidth_bytes_per_cycle,
                        engine.clock_hz,
                    )
                )
        if not values:
            return None
        bandwidths = {value for _, value, _ in values}
        clocks = {clock for _, _, clock in values}
        if len(bandwidths) != 1 or len(clocks) != 1:
            detail = "; ".join(
                f"{name} at {value:g} bytes per cycle of {clock} Hz"
                for name, value, clock in values
            )
            raise ValueError(
                f"port {config.port_id!r} carries mechanisms whose ceilings "
                f"disagree, so the port has no single ceiling: {detail}. Declare "
                "one port per direction instead of averaging two measured rates"
            )
        return GpuPortCeiling(
            bytes_per_cycle=values[0][1],
            clock_hz=values[0][2],
            provenance=GpuPortCeilingProvenance.CALIBRATION_DERIVED,
            source=(
                f"{self.base_architecture.profile_id}/{calibration.calibration_id}"
            ),
        )

    def _copy_engine(self, config: GpuPortConfig) -> CopyEngineProfile:
        assert config.copy_engine_id is not None
        try:
            return self.base_architecture.copy_engine(config.copy_engine_id)
        except KeyError:
            declared = (
                ", ".join(
                    engine.engine_id for engine in self.base_architecture.copy_engines
                )
                or "none"
            )
            raise ValueError(
                f"port {config.port_id!r} names copy engine "
                f"{config.copy_engine_id!r}, which architecture "
                f"{self.base_architecture.profile_id!r} does not declare; it "
                f"declares: {declared}"
            ) from None

    def _effective_architecture(self) -> GpuArchitectureProfile:
        overrides = tuple(
            port
            for port in sorted(self._ports, key=lambda item: item.port_id)
            if port.applicability is GpuPortApplicability.APPLICABLE
            and port.config.declared_ceiling is not None
        )
        if not overrides:
            # Object identity, not equality: a port that carries the current
            # effective parameters must reproduce every accepted artifact byte
            # for byte, and identity is what makes that exact.
            return self.base_architecture
        architecture = self.base_architecture
        calibration = architecture.calibration
        nvlink = calibration.nvlink
        engines = {engine.engine_id: engine for engine in calibration.copy_engines}
        suffix_parts: list[str] = []
        for port in overrides:
            ceiling = port.ceiling
            assert ceiling is not None
            suffix_parts.append(f"{port.port_id}@{ceiling.bytes_per_cycle:g}bpc")
            if port.advertises(GpuPortCapability.PEER_STORE_EGRESS):
                assert nvlink is not None
                nvlink = replace(
                    nvlink,
                    bandwidth_bytes_per_cycle=ceiling.bytes_per_cycle,
                )
            if port.advertises(GpuPortCapability.COPY_ENGINE_TRANSFER):
                assert port.copy_engine_id is not None
                engine = engines[port.copy_engine_id]
                engines[port.copy_engine_id] = replace(
                    engine,
                    direction_profiles=tuple(
                        replace(
                            profile,
                            bandwidth_bytes_per_cycle=ceiling.bytes_per_cycle,
                        )
                        if profile.direction in port.copy_directions
                        else profile
                        for profile in engine.direction_profiles
                    ),
                )
        suffix = "+" + "+".join(suffix_parts)
        provenance = replace(
            calibration.provenance,
            source=(
                f"{calibration.provenance.source} (rescoped by "
                f"{GPU_DEVICE_IMPLEMENTATION} port ceiling declaration)"
            ),
        )
        profile_id = f"{architecture.profile_id}{suffix}"
        derived_calibration = replace(
            calibration,
            calibration_id=f"{calibration.calibration_id}{suffix}",
            target_architecture_profile_id=profile_id,
            provenance=provenance,
            nvlink=nvlink,
            copy_engines=tuple(
                engines[engine.engine_id] for engine in calibration.copy_engines
            ),
        )
        return replace(
            architecture,
            profile_id=profile_id,
            calibration=derived_calibration,
        )


def default_gpu_device_config(
    architecture: GpuArchitectureProfile,
    *,
    device_id: str | None = None,
    host_protocol: GpuPortProtocol = GpuPortProtocol.PCIE,
    peer_protocol: GpuPortProtocol = GpuPortProtocol.NVLINK,
    capabilities: Iterable[GpuPortCapability] = (),
) -> GpuDeviceConfig:
    """Return the port set an architecture's own mechanisms already imply.

    One port per mechanism the calibration declares, each reading its ceiling
    out of that mechanism: the host link is split into one ingress and one
    egress port because the measured host link is not symmetric everywhere, the
    peer copy direction gets its own egress port, and the SM-store egress
    cursor gets one. ``device_to_device`` copies stay inside one GPU and cross
    no port, so no port is created for them. ``capabilities`` adds capabilities
    to every created port, which is how a study opts into
    ``CEILING_OVERRIDE``.
    """

    if not isinstance(architecture, GpuArchitectureProfile):
        raise TypeError("architecture must be a GpuArchitectureProfile")
    extra = tuple(capabilities)
    for capability in extra:
        _require_enum("capability", capability, GpuPortCapability)
    calibration = architecture.calibration
    ports: list[GpuPortConfig] = []
    host_spec = (
        (CopyDirection.HOST_TO_DEVICE, GpuPortDirection.INGRESS, "ingress"),
        (CopyDirection.DEVICE_TO_HOST, GpuPortDirection.EGRESS, "egress"),
    )
    for engine in calibration.copy_engines:
        for copy_direction, port_direction, label in host_spec:
            if copy_direction in engine.directions:
                ports.append(
                    GpuPortConfig(
                        port_id=f"{host_protocol.value}-host-{label}-{engine.engine_id}",
                        protocol=host_protocol,
                        role=GpuPortRole.HOST_LINK,
                        direction=port_direction,
                        capabilities=(
                            GpuPortCapability.COPY_ENGINE_TRANSFER,
                            *extra,
                        ),
                        copy_engine_id=engine.engine_id,
                        copy_directions=(copy_direction,),
                    )
                )
        if CopyDirection.PEER_TO_PEER in engine.directions:
            ports.append(
                GpuPortConfig(
                    port_id=f"{peer_protocol.value}-peer-copy-{engine.engine_id}",
                    protocol=peer_protocol,
                    role=GpuPortRole.PEER_LINK,
                    direction=GpuPortDirection.EGRESS,
                    capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER, *extra),
                    copy_engine_id=engine.engine_id,
                    copy_directions=(CopyDirection.PEER_TO_PEER,),
                )
            )
    if calibration.nvlink is not None:
        ports.append(
            GpuPortConfig(
                port_id=f"{peer_protocol.value}-peer-store",
                protocol=peer_protocol,
                role=GpuPortRole.PEER_LINK,
                direction=GpuPortDirection.EGRESS,
                capabilities=(GpuPortCapability.PEER_STORE_EGRESS, *extra),
            )
        )
    return GpuDeviceConfig(
        device_id=device_id or f"{architecture.profile_id}-device",
        ports=tuple(ports),
    )


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


__all__ = [
    "GPU_DEVICE_CONFIG_VERSION",
    "GPU_DEVICE_IMPLEMENTATION",
    "GPU_PORT_CONFIG_VERSION",
    "MECHANISM_CAPABILITIES",
    "TRANSPORT_CONTROL_CAPABILITIES",
    "GpuDevice",
    "GpuDeviceConfig",
    "GpuPort",
    "GpuPortApplicability",
    "GpuPortCapability",
    "GpuPortCeiling",
    "GpuPortCeilingProvenance",
    "GpuPortConfig",
    "GpuPortDirection",
    "GpuPortProtocol",
    "GpuPortReport",
    "GpuPortRole",
    "default_gpu_device_config",
]
