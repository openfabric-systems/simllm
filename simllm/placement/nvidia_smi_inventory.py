"""Strict readers for the nvidia-smi inventory captured beside an NCCL dump.

NCCL's topology dump names the NVSwitch chips a GPU reaches and how many lanes
land on each, but not which switch port each lane uses. ``nvidia-smi nvlink
-R`` does: for every link of every GPU it prints the remote device's PCI bus
id and the remote link index, which on an NVSwitch board is the switch port.
The readers here turn the blocks of a captured inventory into typed tables
and refuse anything they would otherwise have to guess:

- :func:`read_inventory_sections` splits a capture file at its
  ``--- <title> ---`` headers;
- :func:`read_nvlink_remote_ports` reads the ``nvidia-smi nvlink -R`` block
  into a per-GPU, per-link ``(remote bus id, remote port)`` table, refusing the
  virtual fabric address a container prints when the switches are hidden;
- :func:`read_pci_device_list` reads a ``<bus id> class=... vendor=...
  device=... numa=...`` device list, such as the class ``0x0680`` block;
- :func:`read_module_ids` reads the ``Minor Number`` and ``Module Id`` lines
  of an ``nvidia-smi -q`` block;
- :func:`captured_switch_ports` joins the link table to the NVSwitch device
  list, refusing a link whose remote device is not a listed switch, and
  returns the table :func:`~simllm.placement.nccl_topology.captured_switched_node`
  binds preset switch ports with.

Bus ids are normalized to NCCL's spelling, a four-digit lowercase domain.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: The remote device nvidia-smi prints when the NVSwitch side is not observable.
VIRTUAL_REMOTE_DEVICE = "ffff:ff:ff.0"
#: PCI class prefix of an NVSwitch (bridge, other) in a class 0x0680 device list.
NVSWITCH_PCI_CLASS = "0x068000"

_SECTION = re.compile(r"--- (.+) ---")
_SMI_BUS_ID = re.compile(r"([0-9A-Fa-f]{4,8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-9A-Fa-f])")
_GPU_HEADING = re.compile(r"GPU ([0-9]+): (.+)")
_REMOTE_LINK = re.compile(r"\s*Link ([0-9]+): Remote Device (\S+): Link ([0-9]+)\s*")
_DEVICE_LINE = re.compile(
    r"(\S+)(?: class=(0x[0-9a-f]{6}))? vendor=(0x[0-9a-f]{4}) device=(0x[0-9a-f]{4})"
    r"(?: numa=(-?[0-9]+))?\s*"
)
_QUERY_LINE = re.compile(r"\s*([A-Za-z][A-Za-z0-9 ]*?)\s*:\s*(.*?)\s*")


def _refuse(message: str) -> ValueError:
    return ValueError(f"nvidia-smi inventory refused: {message}")


def normalize_bus_id(value: str) -> str:
    """``00000000:C5:00.0`` or ``0000:c5:00.0`` to ``0000:c5:00.0``."""

    match = _SMI_BUS_ID.fullmatch(value)
    if match is None:
        raise _refuse(f"{value!r} is not a PCI bus id")
    domain = int(match.group(1), 16)
    if domain > 0xFFFF:
        return VIRTUAL_REMOTE_DEVICE if value.upper() == "FFFFFFFF:FF:FF.0" else _raise_domain(value)
    return f"{domain:04x}:{match.group(2)}:{match.group(3)}.{match.group(4)}".lower()


def _raise_domain(value: str) -> str:
    raise _refuse(f"{value!r} has a PCI domain wider than sixteen bits")


def read_inventory_sections(text: str) -> dict[str, tuple[str, ...]]:
    """Split a capture file into its ``--- <title> ---`` blocks, in file order."""

    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        header = _SECTION.fullmatch(line)
        if header is not None:
            current = header.group(1)
            if current in sections:
                raise _refuse(f"section {current!r} appears twice")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {title: tuple(lines) for title, lines in sections.items()}


def read_nvlink_remote_ports(lines: Iterable[str]) -> dict[int, tuple[tuple[str, int], ...]]:
    """Read ``nvidia-smi nvlink -R`` into ``{gpu index: ((bus id, port), ...)}``.

    GPU indices must run 0, 1, 2, ... and each GPU's link indices must run
    0, 1, 2, ... in order, so position ``k`` of a GPU's tuple is its link ``k``.
    """

    table: dict[int, list[tuple[str, int]]] = {}
    current = None
    for line in lines:
        if not line.strip():
            continue
        heading = _GPU_HEADING.fullmatch(line)
        if heading is not None:
            current = int(heading.group(1))
            if current != len(table):
                raise _refuse(f"GPU {current} heading is out of order in the nvlink -R block")
            table[current] = []
            continue
        link = _REMOTE_LINK.fullmatch(line)
        if link is None or current is None:
            raise _refuse(f"unreadable nvlink -R line {line.strip()!r}")
        index, device, port = int(link.group(1)), link.group(2), int(link.group(3))
        if index != len(table[current]):
            raise _refuse(f"GPU {current} link {index} is out of order in the nvlink -R block")
        busid = normalize_bus_id(device)
        if busid == VIRTUAL_REMOTE_DEVICE:
            raise _refuse(
                f"GPU {current} link {index} names the virtual fabric address; the switch side "
                "is not observable"
            )
        table[current].append((busid, port))
    if not table or any(not rows for rows in table.values()):
        raise _refuse("the nvlink -R block lists no GPU links")
    return {gpu: tuple(rows) for gpu, rows in table.items()}


@dataclass(frozen=True)
class InventoryPciDevice:
    """One row of a captured PCI device list."""

    busid: str
    pci_class: str | None
    vendor: str
    device: str
    numa: int | None


def read_pci_device_list(lines: Iterable[str]) -> tuple[InventoryPciDevice, ...]:
    """Read ``<bus id> [class=...] vendor=... device=... [numa=...]`` lines."""

    rows = []
    for line in lines:
        if not line.strip():
            continue
        match = _DEVICE_LINE.fullmatch(line)
        if match is None:
            raise _refuse(f"unreadable PCI device line {line.strip()!r}")
        rows.append(InventoryPciDevice(
            busid=normalize_bus_id(match.group(1)),
            pci_class=match.group(2),
            vendor=match.group(3),
            device=match.group(4),
            numa=None if match.group(5) is None else int(match.group(5)),
        ))
    busids = [row.busid for row in rows]
    if len(set(busids)) != len(busids):
        raise _refuse("a PCI device list repeats a bus id")
    return tuple(rows)


def read_module_ids(lines: Iterable[str]) -> dict[int, int]:
    """Read ``{Minor Number: Module Id}`` from an ``nvidia-smi -q`` block."""

    records: list[dict[str, list[str]]] = []
    for line in lines:
        match = _QUERY_LINE.fullmatch(line)
        if match is None:
            continue
        key, value = match.groups()
        if key == "Product Name":
            records.append({})
        if records:
            records[-1].setdefault(key, []).append(value)
    modules: dict[int, int] = {}
    for record in records:
        minors, ids = record.get("Minor Number", []), record.get("Module Id", [])
        if len(minors) != 1 or len(ids) != 1 or not minors[0].isdigit() or not ids[0].isdigit():
            raise _refuse("a GPU record lacks exactly one integer Minor Number and Module Id")
        if int(minors[0]) in modules:
            raise _refuse(f"Minor Number {minors[0]} appears twice")
        modules[int(minors[0])] = int(ids[0])
    if not modules:
        raise _refuse("the block holds no GPU record")
    return modules


def captured_switch_ports(
    remote_ports: dict[int, Sequence[tuple[str, int]]],
    nvswitches: Iterable[InventoryPciDevice],
) -> dict[int, tuple[tuple[str, int], ...]]:
    """Check every captured link against the listed NVSwitch devices.

    Returns ``{gpu dev: ((switch bus id, switch port), ...)}`` in link-index
    order. A link whose remote device is not a listed NVSwitch is refused, and
    so is a listed device whose class is not an NVSwitch's.
    """

    switches = set()
    for device in nvswitches:
        if device.pci_class not in (None, NVSWITCH_PCI_CLASS):
            raise _refuse(f"listed device {device.busid} has class {device.pci_class}, not an NVSwitch")
        switches.add(device.busid)
    table = {}
    for gpu, rows in sorted(remote_ports.items()):
        for index, (busid, _port) in enumerate(rows):
            if busid not in switches:
                raise _refuse(
                    f"GPU {gpu} link {index} reaches {busid}, which is not a listed NVSwitch"
                )
        table[gpu] = tuple((busid, port) for busid, port in rows)
    return table
