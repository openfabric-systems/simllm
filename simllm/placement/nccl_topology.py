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
          pci(...)                         only under a PCIe switch, class 0x060400
          gpu(dev, sm, rank, gdr)          pci class 0x030200 or 0x030000
            nvlink(target, count, tclass)
          nic                              pci class 0x020000 or 0x020700
            net(name, dev, speed, port, guid, maxconn, gdr)
        nic                                a socket NIC outside any PCI device
          net(name, dev, speed, port, guid, maxconn, gdr)

A GPU's ``<nvlink>`` rows either name peer GPUs by bus id, or are
switch-attached rows of tclass ``0x068000`` with ``count`` lanes each: one
row per NVSwitch chip naming its bus id when the container passes the
switches through, or the single row to the virtual address
``fffffff:ff:ff.0`` NCCL writes when it does not. A GPU mixing peer and
switch rows, or the virtual row and real switch rows, is refused. A socket ``<net>`` may repeat under several
``<cpu>`` elements only when every repetition is attribute-identical.

Attributes NCCL writes beyond the required ones (for example ``familyid`` or
``latency``) are read past, not kept. A PCI device's class must agree with its
child: a GPU sits under a 3D controller (``0x030200``) or VGA controller
(``0x030000``), a NIC under an Ethernet (``0x020000``) or InfiniBand
(``0x020700``) controller, and every other class is refused. A net name, which
becomes part of a fabric NIC identity, must be lowercase letters, digits and
underscores, unique regardless of case.

:func:`captured_fabric_node` gives each GPU the one NIC under its own
``<cpu>`` element and wires
every GPU pair with the bonded NVLink count the dump states. Shapes outside
that rule are refused rather than approximated: a NUMA node with no NIC or
several NICs, a NIC shared by several GPUs or serving none, and a partial
NVLink mesh. The link rate and propagation delay are declared inputs, so the
peer domain keeps ``evidence_class="declared"``; the wiring itself comes from
the dump, which is what a fabric manifest marks with ``source="extracted"``.

:func:`captured_switched_node` joins a switch-attached eight-GPU board to the
declared HGX preset of a named generation instead. It checks the GPU side the
dump can show (eight GPUs, each with the generation's lane count) and takes
the per-chip split from the preset, because the dump collapses every NVSwitch
into one virtual target and cannot show it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from xml.etree import ElementTree

from .dgx import DGX_NVLINK_BUNDLES, dgx_peer_fabric
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
    "cpu": frozenset({"pci", "nic"}),
    "pci": frozenset({"pci", "gpu", "nic"}),
    "gpu": frozenset({"nvlink"}),
    "nic": frozenset({"net"}),
    "nvlink": frozenset(),
    "net": frozenset(),
}

#: PCI class codes a captured GPU may carry: 3D controller and VGA controller.
GPU_PCI_CLASSES = ("0x030200", "0x030000")
#: PCI class codes a captured NIC may carry: Ethernet and InfiniBand controller.
NIC_PCI_CLASSES = ("0x020000", "0x020700")
#: PCI class of a PCIe switch, the only device another ``<pci>`` may nest under.
PCIE_SWITCH_CLASS = "0x060400"
#: The virtual address and class NCCL writes for a GPU's collapsed NVSwitch fabric.
NVSWITCH_VIRTUAL_TARGET = "fffffff:ff:ff.0"
NVSWITCH_TCLASS = "0x068000"

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
    #: bus id of the PCIe switch this device is nested under, or None
    parent_busid: str | None = None


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

    @property
    def switch_attached(self) -> bool:
        """Whether this is the collapsed NVSwitch row rather than a peer row."""

        return self.tclass == NVSWITCH_TCLASS


@dataclass(frozen=True)
class NcclNet:
    """One ``<net>`` element under a ``<nic>``.

    A NIC inside a PCI device carries that device's bus id; a socket NIC listed
    directly under a ``<cpu>`` has ``busid`` None and is placed by ``numaid``.
    """

    busid: str | None
    name: str
    dev: int
    #: link speed as NCCL writes it, in megabits per second
    speed: int
    port: int
    guid: str
    maxconn: int
    gdr: int
    #: NUMA node of the containing ``<cpu>`` element
    numaid: int | None = None

    @property
    def speed_bps(self) -> int:
        return self.speed * 1_000_000


def _refuse(message: str) -> ValueError:
    return ValueError(f"NCCL topology dump refused: {message}")


def _require_rows(value: object, row_type: type, name: str) -> tuple:
    if not isinstance(value, tuple) or any(not isinstance(row, row_type) for row in value):
        raise TypeError(f"NCCL topology {name} must be a tuple of {row_type.__name__}")
    return value


def _net_identity(net: NcclNet) -> tuple:
    return (net.name, net.dev, net.speed, net.port, net.guid, net.maxconn, net.gdr)


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
        by_busid = {device.busid: device for device in self.pci}
        classes = {device.busid: device.pci_class for device in self.pci}
        for device in self.pci:
            if device.numaid not in numa_ids:
                raise _refuse(f"pci {device.busid} names unknown NUMA node {device.numaid}")
            if device.parent_busid is not None:
                parent = by_busid.get(device.parent_busid)
                if (parent is None or parent.pci_class != PCIE_SWITCH_CLASS
                        or parent.numaid != device.numaid):
                    raise _refuse(
                        f"pci {device.busid} is nested under {device.parent_busid}, which is not "
                        f"a PCIe switch (class {PCIE_SWITCH_CLASS}) on its NUMA node"
                    )
        for net in self.nets:
            if net.busid is None and net.numaid not in numa_ids:
                raise _refuse(f"socket net {net.name} names unknown NUMA node {net.numaid}")
        functions = Counter([gpu.busid for gpu in self.gpus]
                            + [net.busid for net in self.nets if net.busid is not None])
        nested = Counter(device.parent_busid for device in self.pci)
        for device in self.pci:
            if nested[device.busid]:
                if functions[device.busid]:
                    raise _refuse(f"pci switch {device.busid} must hold only nested pci devices")
            elif functions[device.busid] != 1:
                raise _refuse(f"pci {device.busid} must hold exactly one gpu or net")
        if set(functions) - set(classes):
            unknown = sorted(set(functions) - set(classes))
            raise _refuse(f"device rows name unknown pci bus ids {', '.join(unknown)}")
        for net in self.nets:
            if net.busid is not None and net.numaid not in (None, by_busid[net.busid].numaid):
                raise _refuse(f"net {net.name} disagrees with the NUMA node of pci {net.busid}")
        gpu_busids = {gpu.busid for gpu in self.gpus}
        for device in self.pci:
            if nested[device.busid]:
                continue
            kind, accepted = (("gpu", GPU_PCI_CLASSES) if device.busid in gpu_busids
                              else ("nic", NIC_PCI_CLASSES))
            if device.pci_class not in accepted:
                raise _refuse(
                    f"pci {device.busid} class {device.pci_class} disagrees with its {kind} "
                    f"child, which requires one of {', '.join(accepted)}"
                )
        _unique([gpu.dev for gpu in self.gpus], "gpu dev")
        _unique([gpu.rank for gpu in self.gpus], "gpu rank")
        groups: dict[str, list[NcclNet]] = {}
        for net in self.nets:
            groups.setdefault(net.name.lower(), []).append(net)
        for key, rows in groups.items():
            if len(rows) == 1:
                continue
            if any(row.busid is not None for row in rows) or len({row.name for row in rows}) != 1:
                raise _refuse(f"duplicate net name (case-insensitive) {key}")
            if len({row.numaid for row in rows}) != len(rows):
                raise _refuse(f"socket net {rows[0].name} repeats under one cpu")
            if len({_net_identity(row) for row in rows}) != 1:
                raise _refuse(
                    f"socket net {rows[0].name} repeats under several cpus with differing "
                    "attributes"
                )
        for net in self.nets:
            if not _NET_NAME.fullmatch(net.name):
                raise _refuse(
                    f"net name {net.name!r} is not lowercase letters, digits and underscores"
                )
        _unique([net.dev for net in self.distinct_nets], "net dev")
        counts: dict[tuple[str, str], int] = {}
        switch_targets: dict[str, list[str]] = {}
        for row in self.nvlinks:
            if row.source_busid not in gpu_busids:
                raise _refuse(f"nvlink source {row.source_busid} is not a gpu")
            if row.target == NVSWITCH_VIRTUAL_TARGET or row.tclass == NVSWITCH_TCLASS:
                if row.tclass != NVSWITCH_TCLASS:
                    raise _refuse(
                        f"gpu {row.source_busid} nvlink row to {row.target} with tclass "
                        f"{row.tclass}; a switch-attached row requires tclass {NVSWITCH_TCLASS}"
                    )
                if row.target in classes and classes[row.target] != NVSWITCH_TCLASS:
                    raise _refuse(
                        f"gpu {row.source_busid} nvlink tclass {row.tclass} disagrees with "
                        f"target {row.target} class {classes[row.target]}"
                    )
                targets = switch_targets.setdefault(row.source_busid, [])
                if row.target in targets:
                    raise _refuse(
                        f"gpu {row.source_busid} repeats its switch-attached nvlink row to "
                        f"{row.target}"
                    )
                targets.append(row.target)
                continue
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
        for source, targets in sorted(switch_targets.items()):
            if NVSWITCH_VIRTUAL_TARGET in targets and len(targets) > 1:
                raise _refuse(
                    f"gpu {source} mixes the virtual NVSwitch row with real switch rows"
                )
        mixed = sorted(set(switch_targets) & {source for source, _ in counts})
        if mixed:
            raise _refuse(f"gpu {mixed[0]} mixes switch-attached and peer-attached nvlink rows")
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

    @property
    def distinct_nets(self) -> tuple[NcclNet, ...]:
        """One row per net name, the first repetition of a socket NIC kept."""

        seen: dict[str, NcclNet] = {}
        for net in self.nets:
            seen.setdefault(net.name, net)
        return tuple(seen.values())

    def pcie_path(self, busid: str) -> tuple[str, ...]:
        """Bus ids from the outermost PCIe switch down to one device."""

        by_busid = {device.busid: device for device in self.pci}
        if busid not in by_busid:
            raise KeyError(f"pci bus id {busid!r} not in NCCL topology dump")
        path = [busid]
        while by_busid[path[-1]].parent_busid is not None:
            path.append(by_busid[path[-1]].parent_busid)
        return tuple(reversed(path))

    def pcie_location(self, busid: str) -> str:
        """``numa-<n>/pci-<switch>/.../pci-<device>`` for one PCI device."""

        return "/".join((f"numa-{self.numa_of(busid)}",
                         *(f"pci-{hop}" for hop in self.pcie_path(busid))))

    def switch_rows(self, gpu_busid: str) -> tuple[NcclNvlink, ...]:
        """A GPU's switch-attached NVLink rows in dump order."""

        return tuple(row for row in self.nvlinks
                     if row.source_busid == gpu_busid and row.switch_attached)

    def switch_lanes(self, gpu_busid: str) -> int:
        """Lanes over all of a GPU's switch-attached rows, zero when it has none."""

        return sum(row.count for row in self.switch_rows(gpu_busid))

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
        socket_attributes: dict[str, dict[str, str]] = {}

        def read_nic(nic_element, numaid: int, busid: str | None) -> None:
            if len(nic_element) != 1:
                holder = f"nic at {busid}" if busid else f"nic under cpu {numaid}"
                raise _refuse(f"{holder} must hold exactly one net element")
            attributes = _check_element(nic_element[0], "nic")
            where = f"net at {busid}" if busid else f"socket net under cpu {numaid}"
            if busid is None:
                previous = socket_attributes.setdefault(attributes["name"], attributes)
                if previous != attributes:
                    raise _refuse(
                        f"socket net {attributes['name']} repeats under several cpus with "
                        "differing attributes"
                    )
            nets.append(NcclNet(
                busid=busid,
                name=_text(attributes, "name", where),
                dev=_decimal(attributes, "dev", where),
                speed=_positive(attributes, "speed", where),
                port=_decimal(attributes, "port", where),
                guid=_hexadecimal(attributes, "guid", where),
                maxconn=_positive(attributes, "maxconn", where),
                gdr=_flag(attributes, "gdr", where),
                numaid=numaid,
            ))

        def read_pci(pci_element, numaid: int, parent_busid: str | None) -> None:
            attributes = _check_element(pci_element, "cpu" if parent_busid is None else "pci")
            busid = _bus_id(attributes, "busid", "pci")
            where = f"pci {busid}"
            pci_class = _hexadecimal(attributes, "class", where)
            pci.append(NcclPciDevice(
                busid=busid,
                numaid=numaid,
                pci_class=pci_class,
                vendor=_hexadecimal(attributes, "vendor", where),
                device=_hexadecimal(attributes, "device", where),
                link_speed=_text(attributes, "link_speed", where),
                link_width=_decimal(attributes, "link_width", where),
                parent_busid=parent_busid,
            ))
            nested = [child for child in pci_element if child.tag == "pci"]
            if nested:
                if pci_class != PCIE_SWITCH_CLASS:
                    raise _refuse(
                        f"{where} of class {pci_class} holds a nested pci; only a PCIe switch "
                        f"(class {PCIE_SWITCH_CLASS}) may"
                    )
                if len(nested) != len(pci_element):
                    raise _refuse(f"pci switch {busid} must hold only nested pci devices")
                for child in nested:
                    read_pci(child, numaid, busid)
                return
            if len(pci_element) != 1:
                raise _refuse(f"{where} must hold exactly one gpu or nic element")
            function = pci_element[0]
            attributes = _check_element(function, "pci")
            if function.tag == "nic":
                read_nic(function, numaid, busid)
                return
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
                target = attributes["target"]
                if target != NVSWITCH_VIRTUAL_TARGET:
                    target = _bus_id(attributes, "target", f"{where} nvlink")
                nvlinks.append(NcclNvlink(
                    source_busid=busid,
                    target=target,
                    count=_positive(attributes, "count", f"{where} nvlink"),
                    tclass=_hexadecimal(attributes, "tclass", f"{where} nvlink"),
                ))

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
            for child in cpu_element:
                if child.tag == "nic":
                    _check_element(child, "cpu")
                    read_nic(child, numaid, None)
                else:
                    read_pci(child, numaid, None)
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


def _rank_map(devs: list[int], global_rank_by_gpu_dev: Mapping[int, int]) -> dict[int, int]:
    if set(global_rank_by_gpu_dev) != set(devs):
        raise ValueError(f"global_rank_by_gpu_dev must name exactly the GPU devs {devs}")
    ranks = {dev: _integer(global_rank_by_gpu_dev[dev], f"global rank of GPU dev {dev}", 0)
             for dev in devs}
    if len(set(ranks.values())) != len(ranks):
        raise ValueError("global_rank_by_gpu_dev assigns one rank to two GPUs")
    return ranks


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
    ranks = _rank_map(devs, global_rank_by_gpu_dev)

    switched = [gpu for gpu in gpus if dump.switch_lanes(gpu.busid)]
    if switched:
        raise ValueError(
            f"GPU dev {switched[0].dev} at {switched[0].busid} is switch-attached; "
            "captured_fabric_node joins a direct NVLink mesh, see captured_switched_node"
        )
    socket = [net for net in dump.nets if net.busid is None]
    if socket:
        raise ValueError(
            f"NIC {socket[0].name} sits outside a PCI device; captured_fabric_node joins "
            "only PCI NICs to GPUs"
        )
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
            pcie_location=dump.pcie_location(gpu.busid),
            nic_id=f"{node_id}:{affine_net[gpu.dev].name}",
        )
        for gpu in gpus
    )
    nic_rows = tuple(
        NicFabricPlacement(
            nic_id=f"{node_id}:{net.name}",
            node_id=node_id,
            fabric_location=dump.pcie_location(net.busid),
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


def captured_switched_node(
    dump: NcclTopologyDump,
    *,
    generation: str,
    node_id: str,
    pool_role: str,
    global_rank_by_gpu_dev: Mapping[int, int],
    propagation_delay_ps: int,
    switch_input_buffer_bytes: int,
    module_id_by_gpu_dev: Mapping[int, int] | None = None,
    switch_ports_by_gpu_dev: Mapping[int, Sequence[tuple[str, int]]] | None = None,
) -> tuple[FabricNodePlacement, PeerFabric]:
    """Join a captured switch-attached eight-GPU board to a declared HGX preset.

    Every GPU's switch-attached rows must carry the preset generation's lanes
    per GPU in total, and the returned
    :class:`~simllm.placement.peer_topology.PeerFabric` is
    :func:`~simllm.placement.dgx.dgx_peer_fabric` for the captured ranks.

    ``switch_ports_by_gpu_dev`` binds the switch side as well. It maps every
    GPU ``dev`` to its links in link-index order, each the ``(switch bus id,
    switch port)`` nvidia-smi reports (see
    :func:`~simllm.placement.nvidia_smi_inventory.captured_switch_ports`). The
    preset's chips, in bundle order, bind to the captured switches in
    ascending bus-id order; every GPU must carry exactly the bundle's width on
    each switch, agreeing with its NCCL switch rows, and no switch port may
    serve two links. Lane ``k`` of a GPU on chip ``s`` is the ``k``-th captured
    link of that GPU on that switch. Switches are then named
    ``<domain>:switch-<busid>`` and switch ports
    ``<domain>:switch-<busid>:port-<n>``; GPU-side names, routes and capacities
    are the preset's. Without the map the switch side keeps its declared names.

    The preset slot of a GPU is its baseboard position. ``module_id_by_gpu_dev``
    maps every GPU ``dev`` to the ``Module Id`` nvidia-smi reports for it, which
    must be a bijection onto 1 through 8; slot ``module_id - 1`` then holds that
    GPU, so every port and link identity follows the physical slot. Without the
    map, slot ``i`` holds the GPU of the ``i``-th smallest ``dev``. GPUs are
    placed at their PCIe path, switches included. A board without a GPU Direct RDMA NIC
    gets declared-absent NICs ``<node_id>:nic-absent-<dev>`` and ``nics=()``;
    a board that carries one is refused, because its NIC selection is not
    modeled here. Every refusal is raised before any schema object is built.
    """

    if not isinstance(dump, NcclTopologyDump):
        raise TypeError("dump must be an NcclTopologyDump")
    if generation not in DGX_NVLINK_BUNDLES:
        raise ValueError(f"generation must be one of {', '.join(DGX_NVLINK_BUNDLES)}")
    _nonblank(node_id, "captured node_id")
    _nonblank(pool_role, "captured pool_role")
    _integer(propagation_delay_ps, "NVLink propagation delay", 0)
    _integer(switch_input_buffer_bytes, "switch input buffer bytes", 1)
    if not isinstance(global_rank_by_gpu_dev, Mapping):
        raise TypeError("global_rank_by_gpu_dev must be a mapping from GPU dev to rank")
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    if len(gpus) != 8:
        raise ValueError(f"a captured switched board requires eight GPUs; the dump holds {len(gpus)}")
    devs = [gpu.dev for gpu in gpus]
    ranks = _rank_map(devs, global_rank_by_gpu_dev)
    slot_order = devs
    if module_id_by_gpu_dev is not None:
        if not isinstance(module_id_by_gpu_dev, Mapping):
            raise TypeError("module_id_by_gpu_dev must be a mapping from GPU dev to module id")
        modules = dict(module_id_by_gpu_dev)
        if (set(modules) != set(devs)
                or any(type(module) is not int for module in modules.values())
                or sorted(modules.values()) != list(range(1, 9))):
            raise ValueError(
                f"module id map not a bijection onto 1..8 over GPU devs {devs}: {modules}"
            )
        slot_order = sorted(devs, key=lambda dev: modules[dev])
    lanes = sum(DGX_NVLINK_BUNDLES[generation])
    for gpu in gpus:
        count = dump.switch_lanes(gpu.busid)
        if count == 0:
            raise ValueError(f"GPU dev {gpu.dev} at {gpu.busid} is not switch-attached")
        if count != lanes:
            raise ValueError(
                f"GPU dev {gpu.dev} at {gpu.busid} has {count} switch-attached NVLink lanes; "
                f"generation {generation} requires {lanes}"
            )
    rdma = [net for net in dump.distinct_nets if net.gdr]
    if rdma:
        raise ValueError(
            f"NIC {rdma[0].name} supports GPU Direct RDMA; NIC selection on a captured "
            "switched board is not modeled"
        )
    switch_ids, switch_port_ids = _captured_switch_side(
        dump, generation, node_id, gpus, slot_order, lanes, switch_ports_by_gpu_dev,
    )

    node = FabricNodePlacement(
        node_id=node_id,
        pool_role=pool_role,
        gpus=tuple(
            GpuFabricPlacement(
                global_rank=ranks[gpu.dev],
                gpu_id=gpu.busid,
                node_id=node_id,
                pcie_location=dump.pcie_location(gpu.busid),
                nic_id=f"{node_id}:nic-absent-{gpu.dev}",
            )
            for gpu in gpus
        ),
        nics=(),
    )
    mesh = dgx_peer_fabric(
        generation,
        node_id=node_id,
        ranks=tuple(ranks[dev] for dev in slot_order),
        propagation_delay_ps=propagation_delay_ps,
        switch_input_buffer_bytes=switch_input_buffer_bytes,
        switch_ids=switch_ids,
        switch_port_ids=switch_port_ids,
    )
    return node, mesh


def _captured_switch_side(
    dump: NcclTopologyDump,
    generation: str,
    node_id: str,
    gpus: list[NcclGpu],
    slot_order: list[int],
    lanes: int,
    switch_ports_by_gpu_dev: Mapping[int, Sequence[tuple[str, int]]] | None,
) -> tuple[tuple[str, ...] | None, dict[tuple[int, int, int], str] | None]:
    """Validate a captured switch-port table and name the preset's switch side."""

    if switch_ports_by_gpu_dev is None:
        return None, None
    if not isinstance(switch_ports_by_gpu_dev, Mapping):
        raise TypeError("switch_ports_by_gpu_dev must be a mapping from GPU dev to links")
    devs = [gpu.dev for gpu in gpus]
    if set(switch_ports_by_gpu_dev) != set(devs):
        raise ValueError(f"switch_ports_by_gpu_dev must name exactly the GPU devs {devs}")
    table: dict[int, list[tuple[str, int]]] = {}
    for dev in devs:
        rows = switch_ports_by_gpu_dev[dev]
        if not isinstance(rows, (list, tuple)) or any(
            not isinstance(row, (list, tuple)) or len(row) != 2 or not isinstance(row[0], str)
            or type(row[1]) is not int or row[1] < 0
            for row in rows
        ):
            raise TypeError(f"captured links of GPU dev {dev} must be (bus id, port) pairs")
        table[dev] = [(row[0], row[1]) for row in rows]
        if len(rows) != lanes:
            raise ValueError(
                f"GPU dev {dev} has {len(rows)} captured NVLink links; generation {generation} "
                f"requires {lanes}"
            )
    bundle = DGX_NVLINK_BUNDLES[generation]
    switches = sorted({busid for rows in table.values() for busid, _ in rows})
    if len(switches) != len(bundle):
        raise ValueError(
            f"generation {generation} binds {len(bundle)} NVSwitch chips; the captured links "
            f"name {len(switches)} switches"
        )
    busid_of = {gpu.dev: gpu.busid for gpu in gpus}
    for dev in devs:
        captured = Counter(busid for busid, _ in table[dev])
        for chip, (switch, width) in enumerate(zip(switches, bundle, strict=True), 1):
            if captured[switch] != width:
                raise ValueError(
                    f"GPU dev {dev} has {captured[switch]} captured lanes on switch {switch}; "
                    f"chip {chip} of generation {generation} requires {width}"
                )
        dumped = {row.target: row.count for row in dump.switch_rows(busid_of[dev])}
        if dumped != dict(captured):
            raise ValueError(
                f"captured links of GPU dev {dev} disagree with its NCCL switch rows {dumped}"
            )
    used = Counter(pair for rows in table.values() for pair in rows)
    repeated = sorted(pair for pair, count in used.items() if count > 1)
    if repeated:
        raise ValueError(
            f"captured switch {repeated[0][0]} port {repeated[0][1]} is used by two links"
        )
    domain = f"{node_id}:hgx-{generation}-8"
    switch_ids = tuple(f"{domain}:switch-{switch}" for switch in switches)
    switch_port_ids: dict[tuple[int, int, int], str] = {}
    for slot, dev in enumerate(slot_order):
        for chip, switch in enumerate(switches, 1):
            ports = [port for busid, port in table[dev] if busid == switch]
            for lane, port in enumerate(ports):
                switch_port_ids[slot, chip, lane] = f"{domain}:switch-{switch}:port-{port}"
    return switch_ids, switch_port_ids
