"""Immutable Ring programs projected from NCCL 2.31.2 buffered primitives.

These records describe source operations and masks, not calibrated GPU cycles.
The runtime supplies the only clock. Work descriptors carry the actual channel
partition; logical channels never stand in for physical links or SMs.
"""

from dataclasses import dataclass

NCCL_SOURCE_COMMIT = "7b83616df3ae082a1f32bb74c27458bfe8153a13"
PROTOCOLS = ("LL", "LL128", "SIMPLE")
LL_CLEAN_MASK = 0x7FFFFFF8
BUFFER_BYTES = {"LL": 524288, "LL128": 4915200, "SIMPLE": 4194304}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class NcclStripe:
    """One warp's useful interval and source-prescribed peer store interval."""

    index: int
    warp: int
    offset_bytes: int
    useful_bytes: int
    encoded_bytes: int
    load_masks: tuple[int, ...]
    local_load_bytes: int
    shared_staging_bytes: int
    shared_tail_load_bytes: int


def protocol_stripes(payload: int, protocol: str, warps: int) -> tuple[NcclStripe, ...]:
    if type(payload) is not int or payload < 0 or payload % 4:
        raise ValueError("stripes require nonnegative float32 bytes")
    if protocol not in PROTOCOLS or type(warps) is not int or warps <= 0:
        raise ValueError("stripes require a supported protocol and positive warps")
    quantum = {"LL": 256, "LL128": 1920, "SIMPLE": 512}[protocol]
    stripes = []
    for index, offset in enumerate(range(0, payload, quantum)):
        useful = min(quantum, payload - offset)
        if protocol == "LL128":
            # prims_ll128.h: loadRegsBegin<8>, aligned input, four groups.
            masks = tuple(sum(1 << lane for lane in range(32)
                              if (lane % 8 != 7 or group % 2 == 0)
                              and (group * 32 - 4 * (group // 2) + lane
                                   - (group % 2) * (lane // 8)) * 16 < useful)
                          for group in range(4))
            load = 16 * sum(mask.bit_count() for mask in masks)
            wire = 2048
            # storeRegs also stages inactive vectors beyond the useful tail.
            staging = 1920 - (useful // 16) * 16
            tail_load = useful % 16
        else:
            unit = 8 if protocol == "LL" else 16
            masks = ((1 << ceil_div(useful, unit)) - 1,)
            load, staging, tail_load = useful, 0, 0
            wire = ceil_div(useful, 8) * 16 if protocol == "LL" else useful
        stripes.append(NcclStripe(index, index % warps, offset, useful, wire,
                                  masks, load, staging, tail_load))
    return tuple(stripes)


@dataclass(frozen=True)
class NcclPrimitive:
    channel: int
    rank_index: int
    ring_loop: int
    stage: int
    slice_index: int
    chunk_index: int
    offset_bytes: int
    useful_bytes: int
    receive: bool
    send: bool
    source_load: bool
    output_store: bool
    reduce: bool
    steps: int
    stripes: tuple[NcclStripe, ...]

    @property
    def encoded_bytes(self) -> int:
        return sum(stripe.encoded_bytes for stripe in self.stripes)


@dataclass(frozen=True)
class NcclRingProgram:
    """An explicit work descriptor, including its realized channel partition."""

    payload_bytes: int
    ranks: tuple[int, ...]
    protocol: str
    channel_bytes: tuple[int, ...]
    warps: int
    connection_mode: str = "buffered"
    communicator: str = "ring"
    source_commit: str = NCCL_SOURCE_COMMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "channel_bytes", tuple(self.channel_bytes))
        if self.source_commit != NCCL_SOURCE_COMMIT:
            raise ValueError("NCCL program source identity is not supported")
        if self.protocol not in PROTOCOLS or self.connection_mode != "buffered":
            raise ValueError("only LL/LL128/Simple buffered peer-write branches are implemented")
        if (type(self.payload_bytes) is not int or self.payload_bytes <= 0 or self.payload_bytes % 4
                or len(self.ranks) not in (2, 4) or len(set(self.ranks)) != len(self.ranks)
                or any(type(rank) is not int or rank < 0 for rank in self.ranks)):
            raise ValueError("Ring requires positive float32 bytes and two/four unique ranks")
        if (not self.channel_bytes or len(self.channel_bytes) > 64
                or any(type(value) is not int or value <= 0 or value % 4 for value in self.channel_bytes)
                or any(value % 16 for value in self.channel_bytes[:-1])
                or sum(self.channel_bytes) != self.payload_bytes):
            raise ValueError("channel partition must conserve bytes with aligned channel starts")
        if type(self.warps) is not int or not 3 <= self.warps <= {"LL": 16, "LL128": 20, "SIMPLE": 17}[self.protocol]:
            raise ValueError("invalid primitive warp count")
        if not isinstance(self.communicator, str) or not self.communicator.strip():
            raise ValueError("communicator identity must be nonblank")

    def primitives(self, channel: int, rank_index: int) -> tuple[NcclPrimitive, ...]:
        """Follow runRing's actual chunk order, including empty tail chunks."""
        width = len(self.ranks)
        if not 0 <= channel < len(self.channel_bytes) or not 0 <= rank_index < width:
            raise ValueError("primitive rank/channel is outside its work descriptor")
        count = self.channel_bytes[channel]
        base = sum(self.channel_bytes[:channel])
        chunk_limit = {"LL": 32768, "LL128": 576000, "SIMPLE": 2097152}[self.protocol]
        result = []
        for loop, offset in enumerate(range(0, count, width * chunk_limit)):
            remaining = count - offset
            chunk = min(chunk_limit, ceil_div(remaining, width * 16) * 16)
            order = [(rank_index + width - 1) % width]
            order += [(rank_index + width - j) % width for j in range(2, width)]
            order += [rank_index]
            order += [(rank_index + width - j) % width for j in range(1, width - 1)]
            order += [(rank_index + 1) % width]
            for stage, chunk_index in enumerate(order):
                useful = max(0, min(chunk, remaining - chunk_index * chunk))
                recv, send = stage > 0, stage < 2 * width - 2
                source_load = stage < width
                output_store = stage >= width - 1
                reduce = recv and source_load
                parts = [useful]
                if self.protocol == "SIMPLE":
                    # genericOp: max(divUp(nelem,16*2)*16, stepSize*2/32).
                    size = max(ceil_div(useful, 128) * 64, 32768)
                    parts = [min(size, useful), max(0, useful - size)]
                slice_offset = 0
                for slice_index, size in enumerate(parts):
                    workers = self.warps - 1 if self.protocol == "SIMPLE" else self.warps
                    result.append(NcclPrimitive(
                        channel, rank_index, loop, stage, slice_index, chunk_index,
                        base + offset + chunk_index * chunk + slice_offset, size,
                        recv, send, source_load, output_store, reduce,
                        2 if self.protocol == "SIMPLE" else 1,
                        protocol_stripes(size, self.protocol, workers),
                    ))
                    slice_offset += size
        return tuple(result)

    @property
    def useful_network_bytes(self) -> int:
        return sum(p.useful_bytes for channel in range(len(self.channel_bytes))
                   for rank in range(len(self.ranks)) for p in self.primitives(channel, rank) if p.send)

    @property
    def encoded_network_bytes(self) -> int:
        return sum(p.encoded_bytes for channel in range(len(self.channel_bytes))
                   for rank in range(len(self.ranks)) for p in self.primitives(channel, rank) if p.send)


def balanced_channels(payload_bytes: int, channels: int) -> tuple[int, ...]:
    """Declared balanced descriptor for interventions, not the NCCL chooser."""
    if type(channels) is not int or channels <= 0 or payload_bytes < 16 * channels:
        raise ValueError("balanced descriptors need at least one aligned cell per channel")
    cells, tail = divmod(payload_bytes, 16)
    return tuple((cells // channels + (index < cells % channels)) * 16
                 + (tail if index == channels - 1 else 0) for index in range(channels))
