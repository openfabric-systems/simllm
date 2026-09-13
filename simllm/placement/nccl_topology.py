"""Captured node inventory from an NCCL topology dump.

NCCL writes the intra-node structure it detected to the file named by
``NCCL_TOPO_DUMP_FILE``: a ``<system version="1">`` XML document with one
``<cpu>`` element per NUMA node, the PCI devices under each, every GPU with
its NVLink rows and every network device with its speed. This module reads
that document strictly and joins one node of it to the repository's existing
schemas: a :class:`~simllm.placement.manifest.FabricNodePlacement` carries the
GPU and NIC inventory with its GPU-to-NIC affinity, and a direct-mesh
:class:`~simllm.placement.peer_topology.PeerFabric` carries the NVLink
attachments. No serialized schema gains a field.

Reading is fail closed. A version other than ``"1"``, an element the schema
below does not name, an absent required attribute, a malformed integer, a
duplicate identity or a cross reference to an unknown device is refused, and
nothing is guessed from what is missing. The accepted structure is::

    system(version)
      cpu(numaid, host_hash, affinity, arch, vendor)
        pci(busid, class, vendor, device, link_speed, link_width)
          gpu(dev, sm, rank, gdr)
            nvlink(target, count, tclass)
          nic
            net(name, dev, speed, port, guid, maxconn, gdr)

Attributes NCCL writes beyond the required ones (for example ``familyid`` or
``latency``) are read past, not kept. A PCI device's class must agree with its
child (``0x030200`` for a GPU, ``0x020000`` for a NIC), and a net name, which
becomes part of a fabric NIC identity, must be lowercase letters, digits and
underscores, unique regardless of case.

The join gives each GPU the one NIC under its own ``<cpu>`` element and wires
every GPU pair with the bonded NVLink count the dump states. Shapes outside
that rule are refused rather than approximated: a NUMA node with no NIC or
several NICs, a NIC shared by several GPUs or serving none, and a partial
NVLink mesh. The link rate and propagation delay are declared inputs, so the
peer domain keeps ``evidence_class="declared"``; the wiring itself comes from
the dump, which is what a fabric manifest marks with ``source="extracted"``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from xml.etree import ElementTree

from .manifest import FabricLink, FabricNodePlacement, GpuFabricPlacement, NicFabricPlacement
from .peer_topology import PeerFabric, PeerPortPlacement, PeerRoute

#: The only system dump version this reader accepts.
NCCL_TOPOLOGY_VERSION = "1"

#: One NVLink3 lane carries 25 GB/s, i.e. 200 Gbit/s, per direction.
DEFAULT_NVLINK_LINK_RATE_BPS = 200_000_000_000

#: Required attributes by element; every element named here is a known one.
_REQUIRED_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "system": ("version",),
    "cpu": ("numaid", "host_hash", "affinity", "arch", "vendor"),
    "pci": ("busid", "class", "vendor", "device", "link_speed", "link_width"),
    "gpu": ("dev", "sm", "rank", "gdr"),
    "nvlink": ("target", "count", "tclass"),
    "nic": (),
    "net": ("name", "dev", "speed", "port", "guid", "maxconn", "gdr"),
}

#: The element each known element may contain.
_CHILD_ELEMENTS: dict[str, frozenset[str]] = {
    "system": frozenset({"cpu"}),
    "cpu": frozenset({"pci"}),
    "pci": frozenset({"gpu", "nic"}),
    "gpu": frozenset({"nvlink"}),
    "nic": frozenset({"net"}),
    "nvlink": frozenset(),
    "net": frozenset(),
}

#: PCI class codes a captured device must carry: 3D controller and Ethernet.
GPU_PCI_CLASS = "0x030200"
NIC_PCI_CLASS = "0x020000"

_NET_NAME = re.compile(r"[a-z0-9_]+")
_DECIMAL = re.compile(r"[0-9]+")
_HEXADECIMAL = re.compile(r"0x[0-9a-f]+")
_BUS_ID = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]")


@dataclass(frozen=True)
class NcclCpu:
    """One ``<cpu>`` element: a NUMA node of the dumping host."""

    numaid: int
    host_hash: str
    affinity: str
    arch: str
    vendor: str


@dataclass(frozen=True)
class NcclPciDevice:
    """One ``<pci>`` element and the NUMA node that contains it."""

    busid: str
    numaid: int
    #: the PCI class code, written ``class`` in the dump
    pci_class: str
    vendor: str
    device: str
    link_speed: str
    link_width: int


@dataclass(frozen=True)
class NcclGpu:
    """One ``<gpu>`` element, identified by the bus id of its PCI device."""

    busid: str
    dev: int
    sm: int
    rank: int
    gdr: int


@dataclass(frozen=True)
class NcclNvlink:
    """One ``<nvlink>`` row: ``count`` bonded links from a GPU to a target."""

    source_busid: str
    target: str
    count: int
    tclass: str


@dataclass(frozen=True)
class NcclNet:
    """One ``<net>`` element under a ``<nic>``, identified by its PCI bus id."""

    busid: str
    name: str
    dev: int
    #: link speed as NCCL writes it, in megabits per second
    speed: int
    port: int
    guid: str
    maxconn: int
    gdr: int

    @property
    def speed_bps(self) -> int:
        return self.speed * 1_000_000


def _refuse(message: str) -> ValueError:
    return ValueError(f"NCCL topology dump refused: {message}")


def _require_rows(value: object, row_type: type, name: str) -> tuple:
    if not isinstance(value, tuple) or any(not isinstance(row, row_type) for row in value):
        raise TypeError(f"NCCL topology {name} must be a tuple of {row_type.__name__}")
    return value


def _unique(values: list, name: str) -> None:
    repeated = sorted(str(value) for value, count in Counter(values).items() if count > 1)
    if repeated:
        raise _refuse(f"duplicate {name} {', '.join(repeated)}")


@dataclass(frozen=True)
class NcclTopologyDump:
    """A strictly read ``NCCL_TOPO_DUMP_FILE`` system document.

    Rows keep file order. Construction checks every identity and cross
    reference, so a dump built by :meth:`parse` or by hand is equally checked.
    """

    version: str
    cpus: tuple[NcclCpu, ...]
    pci: tuple[NcclPciDevice, ...]
    gpus: tuple[NcclGpu, ...]
    nvlinks: tuple[NcclNvlink, ...]
    nets: tuple[NcclNet, ...]

    def __post_init__(self) -> None:
        if self.version != NCCL_TOPOLOGY_VERSION:
            raise _refuse(f"system version {self.version!r} is not {NCCL_TOPOLOGY_VERSION!r}")
        _require_rows(self.cpus, NcclCpu, "cpus")
        _require_rows(self.pci, NcclPciDevice, "pci")
        _require_rows(self.gpus, NcclGpu, "gpus")
        _require_rows(self.nvlinks, NcclNvlink, "nvlinks")
        _require_rows(self.nets, NcclNet, "nets")
        if not self.cpus:
            raise _refuse("the system holds no cpu element")
        _unique([cpu.numaid for cpu in self.cpus], "cpu numaid")
        if len({cpu.host_hash for cpu in self.cpus}) != 1:
            raise _refuse("cpu elements name more than one host_hash")
        numa_ids = {cpu.numaid for cpu in self.cpus}
        _unique([device.busid for device in self.pci], "pci busid")
        classes = {device.busid: device.pci_class for device in self.pci}
        for device in self.pci:
            if device.numaid not in numa_ids:
                raise _refuse(f"pci {device.busid} names unknown NUMA node {device.numaid}")
        functions = Counter([gpu.busid for gpu in self.gpus] + [net.busid for net in self.nets])
        for device in self.pci:
            if functions[device.busid] != 1:
                raise _refuse(f"pci {device.busid} must hold exactly one gpu or net")
        if set(functions) - set(classes):
            unknown = sorted(set(functions) - set(classes))
            raise _refuse(f"device rows name unknown pci bus ids {', '.join(unknown)}")
        gpu_busids = {gpu.busid for gpu in self.gpus}
        for device in self.pci:
            kind, expected = (("gpu", GPU_PCI_CLASS) if device.busid in gpu_busids
                              else ("nic", NIC_PCI_CLASS))
            if device.pci_class != expected:
                raise _refuse(
                    f"pci {device.busid} class {device.pci_class} disagrees with its {kind} "
                    f"child, which requires {expected}"
                )
        _unique([gpu.dev for gpu in self.gpus], "gpu dev")
        _unique([gpu.rank for gpu in self.gpus], "gpu rank")
        _unique([net.name.lower() for net in self.nets], "net name (case-insensitive)")
        for net in self.nets:
            if not _NET_NAME.fullmatch(net.name):
                raise _refuse(
                    f"net name {net.name!r} is not lowercase letters, digits and underscores"
                )
        _unique([net.dev for net in self.nets], "net dev")
        counts: dict[tuple[str, str], int] = {}
        for row in self.nvlinks:
            if row.source_busid not in gpu_busids:
                raise _refuse(f"nvlink source {row.source_busid} is not a gpu")
            if row.target not in gpu_busids:
                raise _refuse(
                    f"gpu {row.source_busid} nvlink target {row.target} is not a known gpu bus id"
                )
            if row.target == row.source_busid:
                raise _refuse(f"gpu {row.source_busid} has an nvlink row to itself")
            if row.tclass != classes[row.target]:
                raise _refuse(
                    f"gpu {row.source_busid} nvlink tclass {row.tclass} disagrees with "
                    f"target {row.target} class {classes[row.target]}"
                )
            key = (row.source_busid, row.target)
            if key in counts:
                raise _refuse(f"gpu {row.source_busid} repeats nvlink target {row.target}")
            counts[key] = row.count
        for (source, target), count in sorted(counts.items()):
            reverse = counts.get((target, source))
            if reverse != count:
                raise _refuse(
                    f"nvlink count {count} from {source} to {target} disagrees with "
                    f"{reverse} in the other direction"
                )

    def numa_of(self, busid: str) -> int:
        """NUMA node of the ``<cpu>`` element holding one PCI device."""

        for device in self.pci:
            if device.busid == busid:
                return device.numaid
        raise KeyError(f"pci bus id {busid!r} not in NCCL topology dump")

    def nvlink_count(self, source_busid: str, target_busid: str) -> int:
        """Bonded NVLink count from one GPU to another, zero when unlinked."""

        for row in self.nvlinks:
            if (row.source_busid, row.target) == (source_busid, target_busid):
                return row.count
        return 0

    @classmethod
    def load(cls, path: str | Path) -> NcclTopologyDump:
        """Read one dump file; see :meth:`parse` for what is refused."""

        return cls.parse(Path(path).read_bytes())

    @classmethod
    def parse(cls, document: str | bytes) -> NcclTopologyDump:
        """Strictly parse the text of one ``<system version="1">`` document."""

        if isinstance(document, str):
            probe = document
        elif isinstance(document, bytes):
            probe = document.decode("utf-8", errors="replace")
        else:
            raise TypeError("NCCL topology document must be text or bytes")
        if "<!DOCTYPE" in probe or "<!ENTITY" in probe:
            raise _refuse("document type and entity declarations are not accepted")
        try:
            root = ElementTree.fromstring(document)
        except ElementTree.ParseError as error:
            raise _refuse(f"malformed XML ({error})") from error
        if root.tag != "system":
            raise _refuse(f"root element <{root.tag}> is not <system>")
        _check_element(root, "document")
        cpus, pci, gpus, nvlinks, nets = [], [], [], [], []
        for cpu_element in root:
            attributes = _check_element(cpu_element, "system")
            numaid = _decimal(attributes, "numaid", "cpu")
            cpus.append(NcclCpu(
                numaid=numaid,
                host_hash=_hexadecimal(attributes, "host_hash", "cpu"),
                affinity=_text(attributes, "affinity", "cpu"),
                arch=_text(attributes, "arch", "cpu"),
                vendor=_text(attributes, "vendor", "cpu"),
            ))
            for pci_element in cpu_element:
                attributes = _check_element(pci_element, "cpu")
                busid = _bus_id(attributes, "busid", "pci")
                where = f"pci {busid}"
                pci.append(NcclPciDevice(
                    busid=busid,
                    numaid=numaid,
                    pci_class=_hexadecimal(attributes, "class", where),
                    vendor=_hexadecimal(attributes, "vendor", where),
                    device=_hexadecimal(attributes, "device", where),
                    link_speed=_text(attributes, "link_speed", where),
                    link_width=_decimal(attributes, "link_width", where),
                ))
                if len(pci_element) != 1:
                    raise _refuse(f"{where} must hold exactly one gpu or nic element")
                function = pci_element[0]
                attributes = _check_element(function, "pci")
                if function.tag == "gpu":
                    where = f"gpu {busid}"
                    gpus.append(NcclGpu(
                        busid=busid,
                        dev=_decimal(attributes, "dev", where),
                        sm=_decimal(attributes, "sm", where),
                        rank=_decimal(attributes, "rank", where),
                        gdr=_flag(attributes, "gdr", where),
                    ))
                    for nvlink_element in function:
                        attributes = _check_element(nvlink_element, "gpu")
                        nvlinks.append(NcclNvlink(
                            source_busid=busid,
                            target=_bus_id(attributes, "target", f"{where} nvlink"),
                            count=_positive(attributes, "count", f"{where} nvlink"),
                            tclass=_hexadecimal(attributes, "tclass", f"{where} nvlink"),
                        ))
                    continue
                if len(function) != 1:
                    raise _refuse(f"nic at {busid} must hold exactly one net element")
                net_element = function[0]
                attributes = _check_element(net_element, "nic")
                where = f"net at {busid}"
                nets.append(NcclNet(
                    busid=busid,
                    name=_text(attributes, "name", where),
                    dev=_decimal(attributes, "dev", where),
                    speed=_positive(attributes, "speed", where),
                    port=_decimal(attributes, "port", where),
                    guid=_hexadecimal(attributes, "guid", where),
                    maxconn=_positive(attributes, "maxconn", where),
                    gdr=_flag(attributes, "gdr", where),
                ))
        return cls(
            version=root.attrib["version"],
            cpus=tuple(cpus),
            pci=tuple(pci),
            gpus=tuple(gpus),
            nvlinks=tuple(nvlinks),
            nets=tuple(nets),
        )


def _check_element(element: ElementTree.Element, parent: str) -> dict[str, str]:
    """Refuse an unknown element, a missing attribute or stray text content."""

    allowed = frozenset({"system"}) if parent == "document" else _CHILD_ELEMENTS[parent]
    if element.tag not in allowed:
        raise _refuse(f"element <{element.tag}> is not accepted under <{parent}>")
    missing = [name for name in _REQUIRED_ATTRIBUTES[element.tag] if name not in element.attrib]
    if missing:
        raise _refuse(f"<{element.tag}> lacks required attribute {', '.join(missing)}")
    if (element.text or "").strip() or (element.tail or "").strip():
        raise _refuse(f"<{element.tag}> carries text content")
    for child in element:
        if child.tag not in _CHILD_ELEMENTS[element.tag]:
            raise _refuse(f"element <{child.tag}> is not accepted under <{element.tag}>")
    return dict(element.attrib)


def _text(attributes: dict[str, str], name: str, where: str) -> str:
    value = attributes[name]
    if not value.strip():
        raise _refuse(f"{where} attribute {name} is blank")
    return value


def _decimal(attributes: dict[str, str], name: str, where: str) -> int:
    value = attributes[name]
    if not _DECIMAL.fullmatch(value):
        raise _refuse(f"{where} attribute {name}={value!r} is not a nonnegative integer")
    return int(value)


def _positive(attributes: dict[str, str], name: str, where: str) -> int:
    value = _decimal(attributes, name, where)
    if value < 1:
        raise _refuse(f"{where} attribute {name} must be positive")
    return value


def _flag(attributes: dict[str, str], name: str, where: str) -> int:
    value = _decimal(attributes, name, where)
    if value not in (0, 1):
        raise _refuse(f"{where} attribute {name} must be 0 or 1")
    return value


def _hexadecimal(attributes: dict[str, str], name: str, where: str) -> str:
    value = attributes[name]
    if not _HEXADECIMAL.fullmatch(value):
        raise _refuse(f"{where} attribute {name}={value!r} is not lowercase hexadecimal")
    return value


def _bus_id(attributes: dict[str, str], name: str, where: str) -> str:
    value = attributes[name]
    if not _BUS_ID.fullmatch(value):
        raise _refuse(f"{where} attribute {name}={value!r} is not a PCI bus id")
    return value


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def captured_fabric_node(
    dump: NcclTopologyDump,
    *,
    node_id: str,
    pool_role: str,
    global_rank_by_gpu_dev: Mapping[int, int],
    nvlink_link_rate_bps: int = DEFAULT_NVLINK_LINK_RATE_BPS,
    nvlink_propagation_delay_ps: int,
    switch_input_buffer_bytes: int = 65536,
) -> tuple[FabricNodePlacement, PeerFabric]:
    """Join one captured node to the fabric and peer-topology schemas.

    ``global_rank_by_gpu_dev`` maps every GPU ``dev`` of the dump to the
    semantic global rank placed on it; the dump's own ``rank`` attribute is the
    dumping communicator's rank and is not used. Every refusal is raised before
    any schema object is built.

    GPUs are emitted in ascending ``dev`` order and NICs in ascending net-name
    order. The peer domain ``<node_id>:nv4`` holds, for each unordered GPU
    pair ``a < b`` by ``dev`` and each of its ``count`` lanes ``k``, the link
    ``<domain>:link-<a>-<b>:lane-<k>`` between the ports
    ``<domain>:gpu-<a>:to-<b>:lane-<k>`` and ``<domain>:gpu-<b>:to-<a>:lane-<k>``,
    plus one route per ordered pair whose paths are the single-link lanes in
    lane order.
    """

    if not isinstance(dump, NcclTopologyDump):
        raise TypeError("dump must be an NcclTopologyDump")
    _nonblank(node_id, "captured node_id")
    _nonblank(pool_role, "captured pool_role")
    _integer(nvlink_link_rate_bps, "NVLink link rate", 1)
    _integer(nvlink_propagation_delay_ps, "NVLink propagation delay", 0)
    _integer(switch_input_buffer_bytes, "switch input buffer bytes", 1)
    if not isinstance(global_rank_by_gpu_dev, Mapping):
        raise TypeError("global_rank_by_gpu_dev must be a mapping from GPU dev to rank")
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    devs = [gpu.dev for gpu in gpus]
    if len(gpus) < 2:
        raise ValueError("a captured NVLink mesh requires at least two GPUs")
    if set(global_rank_by_gpu_dev) != set(devs):
        raise ValueError(f"global_rank_by_gpu_dev must name exactly the GPU devs {devs}")
    ranks = {dev: _integer(global_rank_by_gpu_dev[dev], f"global rank of GPU dev {dev}", 0)
             for dev in devs}
    if len(set(ranks.values())) != len(ranks):
        raise ValueError("global_rank_by_gpu_dev assigns one rank to two GPUs")

    numa = {device.busid: device.numaid for device in dump.pci}
    nets_by_numa: dict[int, list[NcclNet]] = {}
    for net in dump.nets:
        nets_by_numa.setdefault(numa[net.busid], []).append(net)
    gpus_by_numa: dict[int, list[NcclGpu]] = {}
    for gpu in gpus:
        gpus_by_numa.setdefault(numa[gpu.busid], []).append(gpu)
    affine_net: dict[int, NcclNet] = {}
    for gpu in gpus:
        candidates = nets_by_numa.get(numa[gpu.busid], [])
        if len(candidates) != 1:
            raise ValueError(
                f"GPU dev {gpu.dev} at {gpu.busid} has {len(candidates)} NICs under NUMA node "
                f"{numa[gpu.busid]}; the captured-node affinity rule requires exactly one"
            )
        affine_net[gpu.dev] = candidates[0]
    affine_gpu: dict[str, NcclGpu] = {}
    for net in dump.nets:
        served = gpus_by_numa.get(numa[net.busid], [])
        if len(served) != 1:
            raise ValueError(
                f"NIC {net.name} at {net.busid} serves {len(served)} GPUs under NUMA node "
                f"{numa[net.busid]}; the captured-node affinity rule requires exactly one"
            )
        affine_gpu[net.name] = served[0]
    busid = {gpu.dev: gpu.busid for gpu in gpus}
    lanes = {}
    for a, b in combinations(devs, 2):
        count = dump.nvlink_count(busid[a], busid[b])
        if count < 1:
            raise ValueError(
                f"GPU devs {a} and {b} share no NVLink; a partial captured mesh is not modeled"
            )
        lanes[a, b] = lanes[b, a] = count

    gpu_rows = tuple(
        GpuFabricPlacement(
            global_rank=ranks[gpu.dev],
            gpu_id=gpu.busid,
            node_id=node_id,
            pcie_location=f"numa-{numa[gpu.busid]}/pci-{gpu.busid}",
            nic_id=f"{node_id}:{affine_net[gpu.dev].name}",
        )
        for gpu in gpus
    )
    nic_rows = tuple(
        NicFabricPlacement(
            nic_id=f"{node_id}:{net.name}",
            node_id=node_id,
            fabric_location=f"numa-{numa[net.busid]}/pci-{net.busid}",
            affine_gpu_rank=ranks[affine_gpu[net.name].dev],
        )
        for net in sorted(dump.nets, key=lambda net: net.name)
    )
    domain = f"{node_id}:nv4"
    ports: list[PeerPortPlacement] = []
    links: list[FabricLink] = []
    for a, b in combinations(devs, 2):
        for lane in range(lanes[a, b]):
            forward = f"{domain}:gpu-{a}:to-{b}:lane-{lane}"
            backward = f"{domain}:gpu-{b}:to-{a}:lane-{lane}"
            ports.append(PeerPortPlacement(forward, gpu_rank=ranks[a]))
            ports.append(PeerPortPlacement(backward, gpu_rank=ranks[b]))
            links.append(FabricLink(
                f"{domain}:link-{a}-{b}:lane-{lane}", forward, backward,
                nvlink_link_rate_bps, nvlink_propagation_delay_ps,
            ))
    routes = tuple(
        PeerRoute(ranks[source], ranks[destination], tuple(
            (f"{domain}:link-{min(source, destination)}-{max(source, destination)}:lane-{lane}",)
            for lane in range(lanes[source, destination])
        ))
        for source, destination in permutations(devs, 2)
    )
    node = FabricNodePlacement(node_id=node_id, pool_role=pool_role, gpus=gpu_rows, nics=nic_rows)
    mesh = PeerFabric(
        domain_id=domain,
        node_id=node_id,
        ports=tuple(ports),
        links=tuple(links),
        routes=routes,
        protocol="nvlink",
        switch_input_buffer_bytes=switch_input_buffer_bytes,
    )
    return node, mesh
