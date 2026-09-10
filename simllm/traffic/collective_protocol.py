"""NCCL Ring work accounting above the existing physical NVLink transport.

The source projection describes software buffers, not NVLink wire packets.
Service coefficients describe identifiable composite GPU work. In particular,
one measured publication cost does not separately identify a memory round
trip, a fence and a polling loop. All estimates and interval arms are
immutable and deterministic; this module owns no second event calendar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MIB = 1 << 20
PS_PER_SECOND = 10**12
PROTOCOLS = ("LL", "LL128", "SIMPLE")


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, grain: int) -> int:
    return _ceil(a, grain) * grain


def encoded_bytes(payload_bytes: int, protocol: str) -> int:
    """Software-buffer occupancy, including embedded readiness flags."""
    _integer("payload_bytes", payload_bytes)
    if protocol == "LL":
        return _ceil(payload_bytes, 8) * 16
    if protocol == "LL128":
        return _ceil(payload_bytes, 120) * 128
    if protocol == "SIMPLE":
        return payload_bytes
    raise ValueError(f"unsupported NCCL protocol {protocol!r}")


@dataclass(frozen=True)
class NcclRingGeometry:
    """Read-only Ring work, keeping application, flag and counter bytes apart."""

    payload_bytes: int
    width: int
    protocol: str
    tuned_channels: int
    channels: int
    warps: int
    channel_payload_bytes: tuple[int, ...]
    application_endpoint_bytes: int
    encoded_endpoint_floor_bytes: int
    inline_flag_floor_bytes: int
    counter_store_bytes_per_rank: int
    maximum_channel_encoded_bytes: int
    instruction_rounds: int
    nonempty_publications: int
    synchronization_slices: int
    ring_loops: int


def ring_geometry(
    payload_bytes: int,
    width: int,
    protocol: str,
    *,
    maximum_channels: int | None = None,
    fixed_channels: int | None = None,
) -> NcclRingGeometry:
    """Translate the pinned single-collective scheduler and Ring work quanta.

    The default communicator maxima are the measured Merlin meshes, eight
    channels at width two and twenty-four at width four. An explicit fixed
    CTA count models the matching minimum/maximum control. Logical channels
    are not physical links. The last channel retains its partial byte count.
    """
    _integer("payload_bytes", payload_bytes, 16)
    if payload_bytes % 16:
        raise ValueError("the float32 Ring projection requires 16-byte aligned payloads")
    if type(width) is not int or width not in (2, 4):
        raise ValueError("the Merlin Ring projection supports exactly two or four ranks")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported NCCL protocol {protocol!r}")
    maximum = (8 if width == 2 else 24) if maximum_channels is None else maximum_channels
    _integer("maximum_channels", maximum, 1)
    if maximum > 64:
        raise ValueError("maximum_channels exceeds the pinned NCCL limit")
    if fixed_channels is not None:
        _integer("fixed_channels", fixed_channels, 1)
        if fixed_channels > 64:
            raise ValueError("fixed_channels exceeds the pinned NCCL limit")
        maximum = fixed_channels
    threads = 640 if protocol == "LL128" else 512
    threshold = 8 * width if protocol == "LL" else (64 if protocol == "SIMPLE" else 8)
    tuned = maximum
    while payload_bytes < tuned * threads * threshold and tuned >= 2:
        tuned -= 1
    while payload_bytes < tuned * threads * threshold and threads % 128 == 0:
        threads //= 2
    if fixed_channels is not None:
        tuned = fixed_channels
    if protocol == "SIMPLE":
        threads += 32
    threads = max(96, threads)

    # NCCL's scheduler traffic weights are heuristics, not physical wire bytes.
    traffic_per_byte = 8 if protocol == "LL" else 2
    cell_size = _align(_ceil(32768, traffic_per_byte), 16)
    cells = _ceil(payload_bytes, cell_size)
    traffic_per_channel = _align(max(32768, payload_bytes * traffic_per_byte) // tuned, 16)
    cells_per_channel = min(cells, _ceil(traffic_per_channel, cell_size * traffic_per_byte))
    parts = []
    remaining = payload_bytes
    while remaining:
        part = min(remaining, cells_per_channel * cell_size)
        parts.append(part)
        remaining -= part
    if len(parts) > maximum:
        raise ValueError("source channel partition exceeds the communicator maximum")

    hops = 2 * (width - 1)
    chunk_limit = {"LL": 32768, "LL128": 576000, "SIMPLE": 2 * MIB}[protocol]
    largest_work = (0, 0, 0, 0, 0)
    total_counter_bytes = 0
    for part in parts:
        remaining = part
        wire, rounds, publications, slices, loops = 0, 0, 0, 0, 0
        while remaining:
            chunk = min(chunk_limit, _align(_ceil(remaining, width), 16))
            chunk = min(chunk, remaining)
            if protocol == "LL":
                rounds += hops * _ceil(chunk, threads * 8)
                wire += hops * encoded_bytes(chunk, protocol)
                total_counter_bytes += hops * 8
                slices += 2 * width - 1
            elif protocol == "LL128":
                rounds += hops * _ceil(chunk, (threads // 32) * 1920)
                wire += hops * encoded_bytes(chunk, protocol)
                total_counter_bytes += hops * 8
                slices += 2 * width - 1
            else:
                # Two synchronization slices exist even if the second is empty.
                # A nonempty data publication additionally fences system writes.
                slice_bytes = max(_align(_ceil(chunk, 2), 64), 32768)
                nonempty = _ceil(chunk, slice_bytes)
                workers = threads - 32 if threads >= 96 else threads
                for offset in range(0, chunk, slice_bytes):
                    rounds += hops * _ceil(min(slice_bytes, chunk - offset), workers * 16)
                publications += hops * nonempty
                slices += (2 * width - 1) * 2
                wire += hops * chunk
                total_counter_bytes += hops * 2 * 2 * 8
            loops += 1
            remaining = max(0, remaining - width * chunk)
        largest_work = tuple(
            max(a, b)
            for a, b in zip(largest_work, (wire, rounds, publications, slices, loops), strict=True)
        )
    endpoint = hops * (payload_bytes // width)
    encoded = encoded_bytes(endpoint, protocol)
    return NcclRingGeometry(
        payload_bytes,
        width,
        protocol,
        tuned,
        len(parts),
        threads // 32,
        tuple(parts),
        endpoint,
        encoded,
        encoded - endpoint,
        total_counter_bytes,
        *largest_work,
    )


@dataclass(frozen=True)
class NcclProtocolService:
    """Nonnegative effective GPU work costs, each in integer picoseconds."""

    protocol: str
    channel_mib_ps: int
    instruction_round_ps: int
    synchronization_slice_ps: int
    publication_ps: int

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError("unsupported NCCL service protocol")
        for name in (
            "channel_mib_ps",
            "instruction_round_ps",
            "synchronization_slice_ps",
            "publication_ps",
        ):
            _integer(name, getattr(self, name))
        if self.protocol != "SIMPLE" and self.publication_ps:
            raise ValueError("separate data-fence publications belong to Simple")

    def gpu_service_ps(self, geometry: NcclRingGeometry) -> int:
        if geometry.protocol != self.protocol:
            raise ValueError("service and work protocol disagree")
        return (
            _ceil(geometry.maximum_channel_encoded_bytes * self.channel_mib_ps, MIB)
            + geometry.instruction_rounds * self.instruction_round_ps
        )


@dataclass(frozen=True)
class NcclProtocolEstimate:
    geometry: NcclRingGeometry
    physical_floor_ps: int
    reference_ps: int
    method_allowance_ps: int
    central_ps: int
    lower_ps: int
    upper_ps: int
    composite_publication_ps: int
    attributed_rtt_ps: int | None
    residual_poll_fence_ps: int | None


@dataclass(frozen=True)
class NcclRingProtocolModel:
    """Scoped, protocol-discontinuous service with a calibrated uncertainty band.

    `protocol_starts` identifies observed software selections. Source-derived
    channel and work quantization generates the remaining payload structure.
    No per-payload timing anchor is stored. The optional RTT partitions an
    identified composite cost; only a value exceeding that budget adds time.
    That explicit counterfactual never silently fits an additional RTT twice.
    """

    model_id: str
    architecture: str
    width: int
    payload_min_bytes: int
    payload_max_bytes: int
    startup_ps: int
    peer_rate_bytes_per_second: int
    endpoint_rate_bytes_per_second: int
    protocol_starts: tuple[tuple[int, str], ...]
    services: tuple[NcclProtocolService, ...]
    residual_fraction_ppm: int = 0
    protocol_radius_ppm: tuple[tuple[str, int], ...] = ()
    method_intercept_ps: int = 0
    method_physical_fraction_ppm: int = 0
    visibility_rtt_ps: int | None = None
    selection_resolution_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be nonblank")
        if (
            self.architecture not in ("a100", "gh200")
            or type(self.width) is not int
            or self.width not in (2, 4)
        ):
            raise ValueError("model scope requires A100 or GH200 and two or four ranks")
        for name in (
            "payload_min_bytes",
            "payload_max_bytes",
            "peer_rate_bytes_per_second",
            "endpoint_rate_bytes_per_second",
        ):
            _integer(name, getattr(self, name), 1)
        for name in (
            "startup_ps",
            "residual_fraction_ppm",
            "method_intercept_ps",
            "method_physical_fraction_ppm",
            "selection_resolution_bytes",
        ):
            _integer(name, getattr(self, name))
        if self.visibility_rtt_ps is not None:
            _integer("visibility_rtt_ps", self.visibility_rtt_ps)
        if self.payload_min_bytes > self.payload_max_bytes:
            raise ValueError("model payload bounds are inverted")
        object.__setattr__(
            self, "protocol_starts", tuple(tuple(row) for row in self.protocol_starts)
        )
        object.__setattr__(self, "services", tuple(self.services))
        object.__setattr__(
            self, "protocol_radius_ppm", tuple(tuple(row) for row in self.protocol_radius_ppm)
        )
        if self.protocol_radius_ppm and tuple(p for p, _ in self.protocol_radius_ppm) != PROTOCOLS:
            raise ValueError("protocol radii must cover LL, LL128 and SIMPLE in order")
        for _, radius in self.protocol_radius_ppm:
            _integer("protocol radius", radius)
        if not self.protocol_starts or self.protocol_starts[0][0] != self.payload_min_bytes:
            raise ValueError("protocol intervals must begin at the model's lower payload bound")
        previous = -1
        for start, protocol in self.protocol_starts:
            _integer("protocol start", start, 1)
            if start <= previous or start > self.payload_max_bytes or protocol not in PROTOCOLS:
                raise ValueError("invalid protocol selection intervals")
            previous = start
        if any(not isinstance(service, NcclProtocolService) for service in self.services):
            raise TypeError("services must contain NcclProtocolService objects")
        if tuple(service.protocol for service in self.services) != PROTOCOLS:
            raise ValueError("services must contain LL, LL128 and SIMPLE exactly once, in order")

    def protocol(self, payload_bytes: int) -> str:
        _integer("payload_bytes", payload_bytes, 1)
        if not self.payload_min_bytes <= payload_bytes <= self.payload_max_bytes:
            raise ValueError("payload lies outside the calibrated protocol model")
        return next(
            protocol for start, protocol in reversed(self.protocol_starts) if payload_bytes >= start
        )

    def predict(
        self,
        payload_bytes: int,
        *,
        protocol: str | None = None,
        fixed_channels: int | None = None,
    ) -> NcclProtocolEstimate:
        selected = self.protocol(payload_bytes)
        geometry = ring_geometry(
            payload_bytes,
            self.width,
            selected if protocol is None else protocol,
            fixed_channels=fixed_channels,
        )
        service = next(s for s in self.services if s.protocol == geometry.protocol)
        capacity = min(
            self.endpoint_rate_bytes_per_second, (self.width - 1) * self.peer_rate_bytes_per_second
        )
        floor = _ceil(geometry.encoded_endpoint_floor_bytes * PS_PER_SECOND, capacity)
        publication = (
            geometry.nonempty_publications * service.publication_ps
            + geometry.synchronization_slices * service.synchronization_slice_ps
        )
        rtt, residual, extension = None, None, 0
        if self.visibility_rtt_ps is not None:
            rtt = geometry.nonempty_publications * self.visibility_rtt_ps
            residual = max(0, publication - rtt)
            extension = max(0, rtt - publication)
        reference = (
            self.startup_ps + max(floor, service.gpu_service_ps(geometry)) + publication + extension
        )
        method = self.method_intercept_ps + _ceil(floor * self.method_physical_fraction_ppm, 10**6)
        fraction = dict(self.protocol_radius_ppm).get(geometry.protocol, self.residual_fraction_ppm)
        radius = _ceil(reference * fraction, 10**6)
        lower = max(self.startup_ps + floor, reference - radius)
        upper = reference + method + radius
        if protocol is None and self.selection_resolution_bytes:
            for start, alternative in self.protocol_starts[1:]:
                if start - self.selection_resolution_bytes < payload_bytes < start:
                    other = self.predict(
                        payload_bytes, protocol=alternative, fixed_channels=fixed_channels
                    )
                    lower, upper = min(lower, other.lower_ps), max(upper, other.upper_ps)
        return NcclProtocolEstimate(
            geometry,
            floor,
            reference,
            method,
            reference + _ceil(method, 2),
            lower,
            upper,
            publication,
            rtt,
            residual,
        )

    def to_json(self) -> dict:
        return {"schema": "simllm-nccl-ring-protocol-model-v1", **asdict(self)}

    @classmethod
    def from_json(cls, value: dict) -> NcclRingProtocolModel:
        if not isinstance(value, dict):
            raise TypeError("protocol model must be an object")
        payload = dict(value)
        if payload.pop("schema", None) != "simllm-nccl-ring-protocol-model-v1":
            raise ValueError("unsupported protocol model schema")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("protocol model fields do not match the strict schema")
        try:
            payload["services"] = tuple(NcclProtocolService(**row) for row in payload["services"])
            return cls(**payload)
        except (TypeError, KeyError) as error:
            raise ValueError("invalid protocol model fields") from error
