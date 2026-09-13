"""Run the frozen PLACE-6 HGX B200 capture qualification.

The study answers one question: can the eight-GPU switched preset family gain
a `b200` generation whose GPU-side facts equal what a real HGX B200 board
reports through nvidia-smi and NCCL's topology dump, with the reader accepting
the switched dump shape fail closed, the live DGX study path running the new
generation, and every accepted A100 and H100 artifact unchanged?

The input is the rented board fixture under `tests/fixtures/nccl_topology/`.
The frozen expectations are commit `acf0e5e4`, amended before this harness by
the expectations-only commit recorded in `AMENDMENT_COMMIT` (GPU module ids
bind the preset slots) and, after an independent review, by the one recorded in
`AMENDMENT_2_COMMIT` (the join checks each generation's GPU silicon before
binding). Cells G1, G2, G3 and G6 are structural exact guards,
G5 is a rejection control family, and the A100 and H100 preset digests, the
DGX study's tracked results digest and its `--check` run are fatal
by-construction identities.

Cell G4 is live and scored. It reuses the DGX study's own `component`,
`live_config` and `request` helpers at generation `b200` over that study's
component grid and its live cells (tensor-parallel all-reduce and expert
dispatch and combine at widths 2, 4 and 8, attachment rates 12.5 and 25 GB/s
per lane, Python and native switch selection), and runs the `h100` rows the
relations compare against through the same helpers at equal parameters.

Usage:

    python examples/hgx_b200_capture_v1/run_study.py --output <dir> [--library <lib>]
    python examples/hgx_b200_capture_v1/run_study.py --output <dir> --library <lib> --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTATIONS_COMMIT = "acf0e5e4430da97eaf8d5d8f805087f33c942233"
AMENDMENT_COMMIT = "bc1e5c8f9412a723a16865aba4b821894b8e8362"
RESULT_SCHEMA = "simllm-hgx-b200-capture-result-v1"
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-13.json"
AMENDMENT_2_COMMIT = "989482e2ed523922f40392928493d0eb6eab0d7e"
AMENDMENT_2_PATH = STUDY_DIR / "expectations-amendment-2-2026-09-13.json"
RESULTS_PATH = STUDY_DIR / "results.json"
FROZEN_FILES = {
    EXPECTATIONS_COMMIT: ("expectations.md", "expectations.json"),
    AMENDMENT_COMMIT: ("expectations-amendment-2026-09-13.md",
                       "expectations-amendment-2026-09-13.json"),
    AMENDMENT_2_COMMIT: ("expectations-amendment-2-2026-09-13.md",
                         "expectations-amendment-2-2026-09-13.json"),
}
FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "nccl_topology" / "vastai_hgx_b200_8x"
DGX_STUDY = REPOSITORY_ROOT / "examples" / "dgx_nvlink_v1"

NODE_ID = "hgx-b200"
PROPAGATION_DELAY_PS = 1000
SWITCH_INPUT_BUFFER_BYTES = 65536
#: The DGX study's frozen grid, restated because its `run` builds it inline.
RATES = (12_500_000_000, 25_000_000_000)
PAYLOADS = (256, 4096)
CAPACITIES = (272, 65536)
PATTERNS = {
    "pair": ((0, 1),),
    "bidirectional": ((0, 1), (1, 0)),
    "disjoint": ((0, 1), (2, 3), (4, 5), (6, 7)),
    **{f"fanin-{n}": tuple((i, 0) for i in range(1, n + 1)) for n in (1, 3, 7)},
}
FAN_IN_PATTERNS = ("fanin-1", "fanin-3", "fanin-7")
WIDTHS = (2, 4, 8)
KINDS = ("tp", "ep-balanced", "ep-skewed")
#: Declared NVLink 5 bounds of the physical-sanity section: 18 lanes at 50 GB/s.
B200_LANE_PAYLOAD_BYTES_PER_SECOND = 50_000_000_000
B200_GPU_CEILING_BYTES_PER_SECOND = 18 * B200_LANE_PAYLOAD_BYTES_PER_SECOND

#: PCI ID database facts the G3 inventory cross-check compares against.
BOARD_IDS = ["0x5100", "0x5200", "0x6200", "0x6300", "0x7500", "0x7600", "0x8600", "0x8700"]
PART_NUMBER = "692-2G525-0220-500"


def _encode(value):
    if isinstance(value, Fraction):
        return {"denominator": value.denominator, "numerator": value.numerator}
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=_encode) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, check=True,
                          capture_output=True).stdout


def _require_frozen_expectations() -> None:
    """Refuse to run unless both freeze commits precede HEAD and are unchanged."""

    for commit, names in FROZEN_FILES.items():
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                  cwd=REPOSITORY_ROOT, check=False)
        if ancestor.returncode != 0:
            raise SystemExit(f"frozen commit {commit} is not an ancestor of HEAD")
        for name in names:
            relative = (STUDY_DIR / name).relative_to(REPOSITORY_ROOT).as_posix()
            if (STUDY_DIR / name).read_bytes() != _git("show", f"{commit}:{relative}"):
                raise SystemExit(f"{name} differs from frozen commit {commit}")


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes().replace(b"\r\n", b"\n")


def _inventory_section(title: str) -> list[str]:
    lines = _fixture_bytes("node_inventory.txt").decode("utf-8").splitlines()
    start = lines.index(f"--- {title} ---") + 1
    end = next((index for index in range(start, len(lines))
                if lines[index].startswith("--- ") and lines[index].endswith(" ---")),
               len(lines))
    return lines[start:end]


def _smi_query(title: str) -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    for line in _inventory_section(title):
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9 ]*?)\s*:\s*(.*?)\s*", line)
        if match is None:
            continue
        key, value = match.groups()
        if key == "Product Name":
            records.append({})
        if records:
            records[-1].setdefault(key, []).append(value)
    return records


def _smi_bus_id(value: str) -> str:
    domain, rest = value.split(":", 1)
    return f"{int(domain, 16):04x}:{rest.lower()}"


def _module_ids() -> dict[int, int]:
    return {int(row["Minor Number"][0]): int(row["Module Id"][0])
            for row in _smi_query("nvidia-smi -q (GPU board/module ids)")}


def _load_dump():
    from simllm.placement import NcclTopologyDump

    return NcclTopologyDump.load(FIXTURE_DIR / "nccl_topo.xml")


def _join(dump, **overrides):
    from simllm.placement import captured_switched_node

    arguments = {
        "generation": "b200",
        "node_id": NODE_ID,
        "pool_role": "serving",
        "global_rank_by_gpu_dev": {dev: dev for dev in range(8)},
        "propagation_delay_ps": PROPAGATION_DELAY_PS,
        "switch_input_buffer_bytes": SWITCH_INPUT_BUFFER_BYTES,
    }
    arguments.update(overrides)
    return captured_switched_node(dump, **arguments)


def _preset(generation: str):
    from simllm.placement import dgx_peer_fabric

    return dgx_peer_fabric(generation, node_id="node-0", ranks=tuple(range(8)),
                           propagation_delay_ps=PROPAGATION_DELAY_PS,
                           switch_input_buffer_bytes=SWITCH_INPUT_BUFFER_BYTES)


def _canonical_fabric(fabric) -> bytes:
    return json.dumps(asdict(fabric), sort_keys=True, separators=(",", ":")).encode()


def run_g1(dump) -> dict[str, Any]:
    """Cell G1: every literal the parse must reproduce."""

    switches = [device for device in dump.pci if device.pci_class == "0x060400"]
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    devices = {device.busid: device for device in dump.pci}
    graphs = {graph.get("pattern"): graph
              for graph in ElementTree.fromstring(_fixture_bytes("nccl_graph.xml")).iter("graph")}
    socket = dump.distinct_nets
    return {
        "counts": {
            "cpus": len(dump.cpus),
            "distinct_socket_nics": len(socket),
            "gpus": len(gpus),
            "lanes": sorted({row.count for row in dump.nvlinks}),
            "nic_repetitions": len(dump.nets),
            "pcie_switches": len(switches),
            "switch_attached_rows": sum(row.switch_attached for row in dump.nvlinks),
        },
        "gpu": {
            "busid_by_dev": {str(gpu.dev): gpu.busid for gpu in gpus},
            "class": sorted({devices[gpu.busid].pci_class for gpu in gpus}),
            "device": sorted({devices[gpu.busid].device for gpu in gpus}),
            "numa_by_dev": {str(gpu.dev): dump.numa_of(gpu.busid) for gpu in gpus},
            "pcie_switch_by_dev": {str(gpu.dev): devices[gpu.busid].parent_busid for gpu in gpus},
            "sm": sorted({gpu.sm for gpu in gpus}),
            "vendor": sorted({devices[gpu.busid].vendor for gpu in gpus}),
        },
        "pcie_switch": {
            "busids": sorted(device.busid for device in switches),
            "vendor": sorted({device.vendor for device in switches}),
        },
        "nccl_row": sorted({(row.target, row.count, row.tclass) for row in dump.nvlinks}),
        "nic": {
            "gdr": [net.gdr for net in socket],
            "guid": [net.guid for net in socket],
            "name": [net.name for net in socket],
            "numaids": [net.numaid for net in dump.nets],
            "pci_attached": [net.busid is not None for net in dump.nets],
            "port": [net.port for net in socket],
            "speed_mbit": [net.speed for net in socket],
        },
        "nccl": {
            "env": _fixture_bytes("nccl_env.txt").decode("utf-8").splitlines()[0],
            # NCCL graph patterns: 4 is ring, 1 is balanced tree, 5 is NVLS.
            "nvls_channels": int(graphs["5"].get("nchannels")),
            "ring_channels": int(graphs["4"].get("nchannels")),
            "tree_channels": int(graphs["1"].get("nchannels")),
            "typeintra": sorted({graphs[p].get("typeintra") for p in ("4", "1", "5")}),
        },
    }


def run_g2() -> dict[str, Any]:
    """Cell G2: the declared b200 preset."""

    from simllm.placement import FabricNodePlacement, FabricTopologyManifest, GpuFabricPlacement
    from simllm.placement.dgx import DGX_GPU_SILICON

    fabric = _preset("b200")
    chips = sorted({port.switch_id for port in fabric.ports if port.switch_id is not None})
    per_chip = set()
    paths_per_route = set()
    for source, destination in permutations(range(8), 2):
        paths = fabric.paths_between(source, destination)
        paths_per_route.add(len(paths))
        per_chip.update(Counter(path.switch_id for path in paths).values())
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}")
        for rank in range(8)), ())
    try:
        FabricTopologyManifest(nodes=[node], peer_fabrics=(fabric,)).validate()
        valid = True
    except (TypeError, ValueError):
        valid = False
    payload = _canonical_fabric(fabric)
    return {
        "chips": len(chips),
        "silicon": {generation: {"devices": list(devices), "sm": sm}
                    for generation, (devices, sm) in DGX_GPU_SILICON.items()},
        "digest": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        "link_rates_bps": sorted({link.link_rate_bps for link in fabric.links}),
        "links": len(fabric.links),
        "paths_per_chip": sorted(per_chip),
        "paths_per_route": sorted(paths_per_route),
        "ports": len(fabric.ports),
        "routes": len(fabric.routes),
        "validates": valid,
    }


def run_g3(dump) -> dict[str, Any]:
    """Cell G3: the GPU-side binding against nvidia-smi, with module-id slots."""

    from simllm.placement.dgx import DGX_NVLINK_BUNDLES

    lanes = sum(DGX_NVLINK_BUNDLES["b200"])
    fabric = _preset("b200")
    status: dict[int, list[str]] = {}
    current = None
    for line in _inventory_section("nvidia-smi nvlink -s"):
        heading = re.match(r"GPU ([0-9]+): ", line)
        if heading:
            current = int(heading.group(1))
            status[current] = []
        elif line.strip() and current is not None:
            link = re.fullmatch(r"\s*Link [0-9]+: (.+)", line)
            status[current].append(link.group(1) if link else line)
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    matrix_lines = [ansi.sub("", line) for line in _inventory_section("nvidia-smi topo -m")]
    numa_column = matrix_lines[0].split("\t").index("NUMA Affinity")
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in matrix_lines if re.match(r"GPU[0-9]+\t", line)}
    listed = [re.fullmatch(r"GPU ([0-9]+): NVIDIA B200 \(UUID: (GPU-[0-9a-f-]+)\)", line)
              for line in _inventory_section("nvidia-smi -L")]
    full = _smi_query("nvidia-smi -q full")
    modules = _module_ids()
    node, bound = _join(dump, module_id_by_gpu_dev=modules)
    _, unbound = _join(dump)
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    devices = {device.busid: device for device in dump.pci}
    slots: dict[int, set[int]] = {}
    for port in bound.ports:
        if port.gpu_rank is not None:
            slots.setdefault(port.gpu_rank, set()).add(
                int(re.search(r":gpu-([0-9]+):", port.port_id).group(1)))
    owner = {port.port_id: port.gpu_rank for port in bound.ports}
    links_carry_slots = True
    for link in bound.links:
        slot = int(re.search(r":link-([0-9]+)-", link.link_id).group(1))
        gpu_ports = [end for end in (link.endpoint_a, link.endpoint_b) if owner[end] is not None]
        links_carry_slots &= (len(gpu_ports) == 1 and f":gpu-{slot}:" in gpu_ports[0]
                              and slots.get(owner[gpu_ports[0]]) == {slot})
    remote = [re.search(r"Remote Device ([0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+)",
                        line).group(1)
              for line in _inventory_section("nvidia-smi nvlink -R (remote pci bus id per link)")
              if "Remote Device" in line]
    switch_of = [devices[gpu.busid].parent_busid for gpu in gpus]
    return {
        "lanes_per_gpu_equals_active_links": {
            "active_links_by_dev": {str(dev): len(rates) for dev, rates in sorted(status.items())},
            "held": sorted(status) == list(range(8))
            and set(Counter(p.gpu_rank for p in fabric.ports if p.gpu_rank is not None).values())
            == {lanes}
            and all(len(rates) == lanes for rates in status.values()),
            "preset_lanes_per_gpu": lanes,
            "signalling_rates": sorted({rate for rates in status.values() for rate in rates}),
        },
        "paths_per_pair_equals_nv18_and_nccl_count": {
            "held": sorted(rows) == list(range(8)) and all(
                rows[a][1 + b].strip() == "NV18"
                and len(fabric.paths_between(a, b)) == int(rows[a][1 + b].strip()[2:])
                == dump.switch_lanes(gpus[a].busid)
                for a, b in permutations(range(8), 2)
            ),
            "matrix_cells": sorted({rows[a][1 + b].strip() for a, b in permutations(range(8), 2)}),
            "nccl_counts": [row.count for row in dump.nvlinks],
            "ordered_pairs": 56,
        },
        "device_order_and_busids": {
            "held": [match is not None and int(match.group(1)) for match in listed]
            == list(range(8))
            and [int(row["Minor Number"][0]) for row in full] == list(range(8))
            and [row["GPU UUID"][0] for row in full] == [match.group(2) for match in listed]
            and [_smi_bus_id(row["Bus Id"][0]) for row in full]
            == [gpu.gpu_id for gpu in node.gpus] == [gpu.busid for gpu in gpus],
            "join_bus_ids": [gpu.gpu_id for gpu in node.gpus],
        },
        "pcie_pairs_and_numa_split": {
            "held": all(switch_of[dev] == switch_of[dev + 1] for dev in range(0, 8, 2))
            and len(set(switch_of)) == 4
            and Counter(dump.numa_of(gpu.busid) for gpu in gpus) == {0: 4, 1: 4}
            and [int(rows[dev][numa_column]) for dev in range(8)]
            == [dump.numa_of(gpu.busid) for gpu in gpus]
            and [gpu.pcie_location for gpu in node.gpus]
            == [dump.pcie_location(gpu.busid) for gpu in gpus],
            "pcie_locations": [gpu.pcie_location for gpu in node.gpus],
        },
        "distinct_board_ids_one_part_number": {
            "board_ids": [row["Board ID"][0] for row in full],
            "held": [row["Board ID"][0] for row in full] == BOARD_IDS
            and {part for row in full for part in row["Board Part Number"]} == {PART_NUMBER},
            "part_numbers": sorted({part for row in full for part in row["Board Part Number"]}),
        },
        "module_id_slot_binding": {
            "held": slots == {dev: {modules[dev] - 1} for dev in range(8)} and links_carry_slots
            and [p.port_id for p in bound.ports] == [p.port_id for p in unbound.ports]
            and [p.gpu_rank for p in bound.ports] != [p.gpu_rank for p in unbound.ports],
            "links_carry_slots": links_carry_slots,
            "module_id_by_dev": {str(dev): module for dev, module in sorted(modules.items())},
            "slot_by_dev": {str(dev): min(values) for dev, values in sorted(slots.items())},
        },
        "switch_side_not_observable_recorded": {
            "class_0680_devices": sum(bool(line.strip()) for line in
                                      _inventory_section("pci devices class 0x0680 (nvswitch/bridge)")),
            "held": True,
            "nvswitch_devices": sum(bool(line.strip())
                                    for line in _inventory_section("nvswitch devices")),
            "rdma_devices_in_container": sorted(name for name in _inventory_section("net devices")
                                                if name.startswith("mlx5")),
            "remote_device_rows": len(remote),
            "remote_devices": sorted(set(remote)),
        },
    }


def _g5_mutations():
    def seven(root):
        switch = root.find(".//pci[@busid='0000:7a:00.0']")
        switch.remove(switch.find("pci[@busid='0000:87:00.0']"))

    def mixed(root):
        root.find(".//gpu[@dev='0']").append(ElementTree.Element(
            "nvlink", {"target": "0000:52:00.0", "count": "2", "tclass": "0x030200"}))

    return {
        "17 lanes on one gpu": (
            lambda root: root.find(".//gpu[@dev='3']/nvlink").set("count", "17"), {},
            ("GPU dev 3 at 0000:63:00.0 has 17 switch-attached NVLink lanes; "
             "generation b200 requires 18")),
        "seven gpus": (seven, {"global_rank_by_gpu_dev": {dev: dev for dev in range(7)}},
                       "requires eight GPUs; the dump holds 7"),
        "mixed switch and peer rows": (
            mixed, {}, "gpu 0000:51:00.0 mixes switch-attached and peer-attached nvlink rows"),
        "socket nic repeated with differing attributes": (
            lambda root: root.findall("cpu/nic/net")[1].set("speed", "25000"), {},
            "socket net eth0 repeats under several cpus with differing attributes"),
        "nested pci under non-switch": (
            lambda root: root.find(".//pci[@busid='0000:45:00.0']").set("class", "0x030200"), {},
            ("pci 0000:45:00.0 of class 0x030200 holds a nested pci; only a PCIe switch "
             "(class 0x060400) may")),
        "generation a100 against eighteen lanes": (
            None, {"generation": "a100"},
            "has 18 switch-attached NVLink lanes; generation a100 requires 12"),
        "generation h100 against b200 silicon": (
            None, {"generation": "h100"},
            ("GPU dev 0 at 0000:51:00.0 reports device 0x10de:0x2901 with sm 100; "
             "generation h100 requires device 0x2330 or 0x2335 or 0x2339 with sm 90")),
        "module id map not a bijection onto 1..8": (
            None, {"module_id_by_gpu_dev": {**_module_ids(), 7: 4}},
            "module id map not a bijection onto 1..8"),
    }


def run_g5(dump) -> dict[str, Any]:
    """Cell G5: each refusal, raised before any schema object exists."""

    from simllm.placement import NcclTopologyDump, nccl_topology

    rows: dict[str, Any] = {}
    built: list[str] = []
    names = ("FabricNodePlacement", "GpuFabricPlacement", "dgx_peer_fabric")
    originals = {name: getattr(nccl_topology, name) for name in names}

    def sentinel(name):
        def build(*_args, **_kwargs):
            built.append(name)
            raise AssertionError(f"{name} built before the refusal")
        return build

    try:
        for name in names:
            setattr(nccl_topology, name, sentinel(name))
        for label, (change, overrides, expected) in _g5_mutations().items():
            built.clear()
            try:
                if change is None:
                    subject = dump
                else:
                    root = ElementTree.fromstring(_fixture_bytes("nccl_topo.xml"))
                    change(root)
                    subject = NcclTopologyDump.parse(ElementTree.tostring(root, encoding="unicode"))
                _join(subject, **overrides)
            except ValueError as error:
                rows[label] = {"expected_message": expected, "message": str(error),
                               "refused": not built and expected in str(error)}
            except Exception as error:  # noqa: BLE001 - any other type is a wrong refusal
                rows[label] = {"expected_message": expected,
                               "message": f"{type(error).__name__}: {error}", "refused": False}
            else:
                rows[label] = {"expected_message": expected, "message": None, "refused": False}
    finally:
        for name, value in originals.items():
            setattr(nccl_topology, name, value)
    return rows


def _packet_timing(result) -> list[tuple]:
    return sorted((p.source, p.destination, p.payload_bytes, p.wire_bytes, p.released_at_ps,
                   p.tx_started_at_ps, p.tx_finished_at_ps, p.switch_started_at_ps,
                   p.switch_finished_at_ps, p.rx_started_at_ps, p.rx_finished_at_ps,
                   p.visible_at_ps) for p in result.packets)


def _single_packet_ps(rate: int, endpoint: int) -> int:
    def ceil_ps(wire, bytes_per_second):
        return (wire * 10**12 + bytes_per_second - 1) // bytes_per_second
    return (max(ceil_ps(272, endpoint), ceil_ps(272, rate)) + 2 * PROPAGATION_DELAY_PS
            + max(ceil_ps(272, 25_000_000_000), ceil_ps(272, rate)) + ceil_ps(272, endpoint))


def run_g4(output: Path, library: str) -> dict[str, Any]:
    """Cell G4: the DGX study's component grid and live cells at b200."""

    from examples.dgx_nvlink_v1 import run_study as dgx
    from simllm.backends.htsim_nvlink import NvlinkSwitchArbitration

    raw = output / "g4"
    raw.mkdir(parents=True, exist_ok=True)
    guards: dict[str, int] = Counter()
    components = []
    for rate, payload, capacity, policy, pattern in product(
        RATES, PAYLOADS, CAPACITIES, tuple(NvlinkSwitchArbitration), PATTERNS,
    ):
        pairs = PATTERNS[pattern]
        python = dgx.component("b200", rate, payload, pairs, capacity, policy, None)
        native = dgx.component("b200", rate, payload, pairs, capacity, policy, library)
        guards["component_physics"] += 2
        row = {"capacity": capacity, "completion_ps": native.completion_time_ps,
               "pattern": pattern, "payload": payload, "policy": policy.value, "rate": rate,
               "python_native_exact": python == native}
        if pattern == "pair" or pattern in FAN_IN_PATTERNS:
            h100 = dgx.component("h100", rate, payload, pairs, capacity, policy, library)
            guards["component_physics"] += 1
            row["h100_completion_ps"] = h100.completion_time_ps
            if pattern == "pair":
                row["h100_identical_timing"] = _packet_timing(native) == _packet_timing(h100)
        if pattern == "pair" and payload == 256:
            row["single_packet_equation"] = (
                native.completion_time_ps == _single_packet_ps(rate, 450 * 10**9))
        components.append(row)
    isolated_direction = []
    for row in components:
        if row["rate"] == RATES[1] and row["pattern"] in ("pair", "disjoint"):
            slower = next(other for other in components if other["rate"] == RATES[0] and all(
                other[key] == row[key] for key in ("capacity", "pattern", "payload", "policy")))
            isolated_direction.append(row["completion_ps"] <= slower["completion_ps"])
    fast = dgx.component("b200", RATES[1], 4096, ((0, 1),), 272,
                         NvlinkSwitchArbitration.IDENTITY, library)
    delayed = dgx.component("b200", RATES[1], 4096, ((0, 1),), 272,
                            NvlinkSwitchArbitration.IDENTITY, library, processing=100000)

    live = []
    for rate, width, kind in product(RATES, WIDTHS, KINDS):
        runs = {}
        for label, generation, selected in (("python", "b200", None), ("native", "b200", library),
                                            ("h100", "h100", library)):
            name = f"{generation}-{rate}-{width}-{kind}-{label}"
            runs[label] = dgx.request(dgx.live_config(raw / name, generation, rate, width, kind,
                                                      selected))
        (raw / f"b200-{rate}-{width}-{kind}.json").write_bytes(dgx.canonical(runs["native"]))
        (raw / f"h100-{rate}-{width}-{kind}.json").write_bytes(dgx.canonical(runs["h100"]))
        native, h100 = runs["native"], runs["h100"]
        live.append({
            "compute_ps": [step["compute_ps"] for step in native["steps"]],
            "h100_jct_ps": h100["jct_ps"],
            "h100_step_latency_ps": [step["result"]["step_latency_ps"] for step in h100["steps"]],
            "jct_ps": native["jct_ps"],
            "kind": kind,
            "python_native_exact": dgx.canonical(dgx.comparable(runs["python"]))
            == dgx.canonical(dgx.comparable(native)),
            "rate": rate,
            "step_latency_ps": [step["result"]["step_latency_ps"] for step in native["steps"]],
            "tpot_ps": native["tpot_ps"],
            "ttft_ps": native["ttft_ps"],
            "width": width,
        })

    rate_pairs = []
    for width, kind in product(WIDTHS, KINDS):
        slow, quick = (next(row for row in live if (row["rate"], row["width"], row["kind"])
                            == (rate, width, kind)) for rate in RATES)
        rate_pairs.append({"compute_identical": slow["compute_ps"] == quick["compute_ps"],
                           "jct_changed": slow["jct_ps"] != quick["jct_ps"],
                           "kind": kind, "width": width})
    pair_cells = [row for row in components if row["pattern"] == "pair"]
    fan_in_cells = [row for row in components if row["pattern"] in FAN_IN_PATTERNS]
    strict = (
        [{"cell": "component", **{k: row[k] for k in ("rate", "payload", "capacity", "policy",
                                                     "pattern")},
          "b200_ps": row["completion_ps"], "h100_ps": row["h100_completion_ps"]}
         for row in fan_in_cells if row["completion_ps"] > row["h100_completion_ps"]]
        + [{"cell": "live", "kind": row["kind"], "rate": row["rate"], "width": row["width"],
            "b200_step_latency_ps": row["step_latency_ps"],
            "h100_step_latency_ps": row["h100_step_latency_ps"]}
           for row in live if row["step_latency_ps"] != row["h100_step_latency_ps"]]
    )
    return {
        "components": components,
        "credit_backpressure": delayed.completion_time_ps > fast.completion_time_ps,
        "endpoint_ceiling_bytes_per_second": 450 * 10**9,
        "isolated_rate_direction": {"held": all(isolated_direction),
                                    "instances": len(isolated_direction)},
        "live": live,
        "physics_checked_runs": guards["component_physics"],
        "relations": {
            "b200_at_or_above_h100_fan_in_and_live": {
                "fan_in_cells": len(fan_in_cells),
                "held": all(row["completion_ps"] >= row["h100_completion_ps"]
                            for row in fan_in_cells)
                and all(b >= h for row in live
                        for b, h in zip(row["step_latency_ps"], row["h100_step_latency_ps"],
                                        strict=True)),
                "live_cells": len(live),
                "strict_cells": strict,
            },
            "b200_equals_h100_isolated_pair": {
                "cells": len(pair_cells),
                "held": all(row["h100_identical_timing"]
                            and row["completion_ps"] == row["h100_completion_ps"]
                            for row in pair_cells),
            },
            "python_native_exact": {
                "component_cells": len(components),
                "held": all(row["python_native_exact"] for row in (*components, *live)),
                "live_cells": len(live),
            },
            "rate_pair_changes_completion": {
                "changed_pairs": sum(row["jct_changed"] for row in rate_pairs),
                "held": all(row["compute_identical"] for row in rate_pairs)
                and any(row["jct_changed"] for row in rate_pairs),
                "pairs": rate_pairs,
            },
        },
        "single_packet_equation": all(row.get("single_packet_equation", True)
                                      for row in components),
    }


def run_g6(dump, output: Path, library: str) -> dict[str, Any]:
    """Cell G6: wire identity, accepted preset digests and the DGX study's check."""

    from simllm.placement import FabricTopologyManifest

    node, mesh = _join(dump, module_id_by_gpu_dev=_module_ids())
    fabric = FabricTopologyManifest(nodes=[node], peer_fabrics=(mesh,), source="extracted")
    cell = output / "g6"
    cell.mkdir(parents=True, exist_ok=True)
    first = fabric.save(cell / "fabric.json")
    loaded = FabricTopologyManifest.load(first)
    second = loaded.save(cell / "fabric-round-trip.json")
    digests = {}
    for generation in ("a100", "h100"):
        payload = _canonical_fabric(_preset(generation))
        digests[f"dgx_{generation}_peer_fabric_json"] = {
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    check = subprocess.run(
        [sys.executable, str(DGX_STUDY / "run_study.py"), "--library", library,
         "--output", str(output / "dgx-check"), "--check"],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False,
    )
    (cell / "dgx-check.log").write_text(check.stdout + check.stderr, encoding="utf-8")
    return {
        "a100 digest": digests["dgx_a100_peer_fabric_json"],
        "dgx study check reproduces": check.returncode == 0,
        "dgx_results_sha256": hashlib.sha256((DGX_STUDY / "results.json").read_bytes()).hexdigest(),
        "fabric round trip": loaded == fabric and first.read_bytes() == second.read_bytes(),
        "h100 digest": digests["dgx_h100_peer_fabric_json"],
    }


def analyze(cells: dict[str, Any], frozen: dict[str, Any], amendment: dict[str, Any],
            amendment_2: dict[str, Any], fixture_sha256: dict[str, str]) -> dict[str, Any]:
    """Apply every frozen guard, then score the G4 oracle and relations."""

    fatal: list[str] = []
    structural: list[str] = []
    rejection: list[str] = []
    scored: list[str] = []
    capture = frozen["capture"]
    for name, digest in frozen["baseline"]["fixture"].items():
        if name != "directory" and fixture_sha256.get(name) != digest:
            fatal.append(f"fixture digest {name}")

    g1, expected_g1 = cells["g1_parse"], frozen["cells"]["g1_parse"]
    counts = g1["counts"]
    if (counts["cpus"], counts["pcie_switches"], counts["gpus"], counts["switch_attached_rows"],
            counts["lanes"], counts["distinct_socket_nics"], counts["nic_repetitions"]) != (
            expected_g1["cpus"], expected_g1["pcie_switches"], expected_g1["gpus"],
            expected_g1["switch_attached_rows"], [expected_g1["lanes"]],
            expected_g1["distinct_socket_nics"], expected_g1["nic_repetitions"]):
        structural.append("g1: element counts")
    gpu = capture["gpu"]
    if (g1["gpu"]["busid_by_dev"], g1["gpu"]["numa_by_dev"], g1["gpu"]["pcie_switch_by_dev"]) != (
            gpu["busid_by_dev"], gpu["numa_by_dev"], gpu["pcie_switch_by_dev"]):
        structural.append("g1: GPU placement literals")
    if (g1["gpu"]["vendor"], g1["gpu"]["device"], g1["gpu"]["sm"], g1["gpu"]["class"]) != (
            [gpu["vendor"]], [gpu["device"]], [gpu["sm"]], ["0x030200"]):
        structural.append("g1: GPU identity literals")
    if g1["pcie_switch"] != {"busids": sorted(set(gpu["pcie_switch_by_dev"].values())),
                             "vendor": ["0x1000"]}:
        structural.append("g1: PCIe switch literals")
    row = capture["nvlink"]["nccl_row"]
    if g1["nccl_row"] != [[row["target"], row["count"], row["tclass"]]]:
        structural.append("g1: NCCL switch-attached row")
    nic = capture["nic"]
    if (g1["nic"]["name"], g1["nic"]["speed_mbit"], g1["nic"]["gdr"], g1["nic"]["guid"],
            g1["nic"]["port"], len(g1["nic"]["numaids"]), g1["nic"]["pci_attached"]) != (
            [nic["name"]], [nic["speed_mbit"]], [nic["gdr"]], ["0x0"], [0],
            nic["repeated_under_cpus"], [False] * nic["repeated_under_cpus"]):
        structural.append("g1: socket NIC literals")
    nccl = capture["nccl"]
    if (g1["nccl"]["ring_channels"], g1["nccl"]["tree_channels"], g1["nccl"]["nvls_channels"],
            g1["nccl"]["typeintra"]) != (nccl["ring_channels"], nccl["tree_channels"],
                                         nccl["nvls_channels"], ["NVL"]):
        structural.append("g1: NCCL graph channels")
    if f"nccl ({nccl['version'].replace('.', ', ')})" not in g1["nccl"]["env"]:
        structural.append("g1: NCCL version")

    g2, expected_g2, preset = cells["g2_preset"], frozen["cells"]["g2_preset"], frozen["preset"]
    if (g2["chips"], g2["links"], g2["ports"], g2["routes"], g2["paths_per_route"],
            g2["paths_per_chip"], g2["link_rates_bps"], g2["validates"]) != (
            expected_g2["chips"], expected_g2["links"], expected_g2["ports"],
            expected_g2["routes"], [expected_g2["paths_per_route"]],
            [expected_g2["paths_per_chip"]], [expected_g2["link_rate_bps"]], True):
        structural.append("g2: preset structure")
    if preset["link_payload_rate_bps"] > 8 * B200_LANE_PAYLOAD_BYTES_PER_SECOND:
        structural.append("g2: declared lane rate above the NVLink 5 payload bound")
    if g2["silicon"] != amendment_2["silicon"]:
        structural.append("g2: silicon table disagrees with the second amendment")

    g3 = cells["g3_gpu_side"]
    for check in frozen["cells"]["g3_gpu_side"]:
        if g3.get(check, {}).get("held") is not True:
            structural.append(f"g3: {check}")
    if g3["module_id_slot_binding"]["held"] is not True:
        structural.append("g3: module id slot binding")
    corrected = amendment["corrected"]
    if (g3["module_id_slot_binding"]["module_id_by_dev"], g3["module_id_slot_binding"]["slot_by_dev"]) != (
            corrected["module_id_by_dev"], corrected["slot_by_dev"]):
        structural.append("g3: amended module ids or slots")
    unobservable = g3["switch_side_not_observable_recorded"]
    if (unobservable["remote_devices"], unobservable["remote_device_rows"],
            unobservable["class_0680_devices"], unobservable["rdma_devices_in_container"]) != (
            [capture["nvlink"]["remote_device"]], 8 * 18, capture["nvlink"]["switch_pci_devices_visible"],
            corrected["rdma_devices_in_container"]):
        structural.append("g3: switch-side evidence")
    if g3["lanes_per_gpu_equals_active_links"]["signalling_rates"] != [
            f"{capture['nvlink']['signalling_gb_per_s']} GB/s"]:
        structural.append("g3: signalling rate")

    g4 = cells["g4_live"]
    if g4["physics_checked_runs"] < 1 or g4["single_packet_equation"] is not True:
        fatal.append("g4: packet physics or single-packet equation")
    if g4["isolated_rate_direction"]["held"] is not True or g4["credit_backpressure"] is not True:
        fatal.append("g4: isolated rate direction or credit backpressure")
    if g4["endpoint_ceiling_bytes_per_second"] > B200_GPU_CEILING_BYTES_PER_SECOND:
        fatal.append("g4: endpoint ceiling above the 900 GB/s NVLink 5 bound")
    expected_g4 = frozen["cells"]["g4_live"]
    if sorted({row["width"] for row in g4["live"]}) != expected_g4["widths"] or sorted(
            {row["rate"] for row in g4["live"]}) != expected_g4["rates_bytes_per_second"]:
        fatal.append("g4: live grid")
    for relation in expected_g4["relations"]:
        if g4["relations"][relation]["held"] is not True:
            scored.append(f"g4: relation {relation}")

    g5 = cells["g5_refusals"]
    g5_labels = [*frozen["cells"]["g5_refusals"], amendment["added"]["g5_refusal"]]
    for label in g5_labels:
        # The second amendment turns this control's h100 clause into a silicon refusal.
        if label == "generation a100 against eighteen lanes" and g5.get(
                "generation h100 against b200 silicon", {}).get("refused") is not True:
            rejection.append("g5: generation h100 was not refused on silicon")
        if g5.get(label, {}).get("refused") is not True:
            rejection.append(f"g5: {label} was not refused before any schema object")

    g6 = cells["g6_off_path"]
    artifacts = frozen["baseline"]["artifacts"]
    for generation in ("a100", "h100"):
        expected = artifacts[f"dgx_{generation}_peer_fabric_json"]
        if g6[f"{generation} digest"] != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            fatal.append(f"g6: {generation} digest")
    if g6["dgx_results_sha256"] != frozen["baseline"]["files"]["examples/dgx_nvlink_v1/results.json"]:
        fatal.append("g6: DGX study results digest")
    if g6["dgx study check reproduces"] is not True:
        fatal.append("g6: DGX study check")
    if g6["fabric round trip"] is not True:
        structural.append("g6: fabric round trip")

    findings = [*fatal, *structural, *rejection, *scored]
    status = "VOID" if (fatal or structural or rejection) else ("FAIL" if scored else "PASS")
    relations = g4["relations"]
    return {
        "evidence": {
            "behavioral_relation_families": {
                name: {"held": relations[name]["held"]}
                for name in ("rate_pair_changes_completion", "b200_equals_h100_isolated_pair",
                             "b200_at_or_above_h100_fan_in_and_live")
            },
            "exact_oracle_family": {
                "held": relations["python_native_exact"]["held"],
                "instances": relations["python_native_exact"]["component_cells"]
                + relations["python_native_exact"]["live_cells"],
            },
            "fatal_compatibility_digests": 3,
            "rejection_control_family": {
                "controls": len(frozen["cells"]["g5_refusals"]) + 1,
                "refused": sum(g5.get(label, {}).get("refused") is True for label in g5_labels),
            },
            "structural_guard_cells": len(frozen["evidence"]["structural_guard_cells"]),
        },
        "findings": findings,
        "status": status,
    }


def _build_library(output: Path) -> str:
    if not shutil.which("cmake"):
        raise SystemExit("building the native switch library requires CMake; pass --library")
    build = output / "nvswitch-build"
    subprocess.run(["cmake", "-S", str(REPOSITORY_ROOT / "simllm/backends/nvswitch"), "-B",
                    str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"],
                   check=True, capture_output=True)
    name = "simllm_nvswitch.dll" if sys.platform == "win32" else (
        "libsimllm_nvswitch.dylib" if sys.platform == "darwin" else "libsimllm_nvswitch.so")
    return str(next(build.rglob(name)))


def run_study(output: Path, library: str) -> dict[str, Any]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    amendment_2 = json.loads(AMENDMENT_2_PATH.read_text(encoding="utf-8"))
    dump = _load_dump()
    cells = {
        "g1_parse": run_g1(dump),
        "g2_preset": run_g2(),
        "g3_gpu_side": run_g3(dump),
        "g4_live": run_g4(output, library),
        "g5_refusals": run_g5(dump),
        "g6_off_path": run_g6(dump, output, library),
    }
    cells = json.loads(_json_bytes(cells))
    fixture_sha256 = {name: hashlib.sha256(_fixture_bytes(name)).hexdigest()
                      for name in sorted(frozen["baseline"]["fixture"]) if name != "directory"}
    analysis = analyze(cells, frozen, amendment, amendment_2, fixture_sha256)
    summary = {
        "amendment_2_commit": AMENDMENT_2_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "cells": cells,
        "evidence": analysis["evidence"],
        "expectations_commit": EXPECTATIONS_COMMIT,
        "findings": analysis["findings"],
        "fixture_sha256": fixture_sha256,
        "implementation_commit": _git("rev-parse", "HEAD").decode().strip(),
        "schema": RESULT_SCHEMA,
        "status": analysis["status"],
    }
    text = json.dumps(summary, sort_keys=True)
    for machine_path in {str(output), str(REPOSITORY_ROOT), str(Path.home()), library}:
        if machine_path and machine_path in text:
            summary["findings"].append("a machine path reached the tracked summary")
            summary["status"] = "VOID"
            break
    return summary


def comparable(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop the one key that legitimately moves between reproducing runs."""

    view = copy.deepcopy(summary)
    view.pop("implementation_commit", None)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PLACE-6 HGX B200 capture study")
    parser.add_argument("--output", type=Path, required=True,
                        help="external evidence directory, absent or empty")
    parser.add_argument("--library", help="built native switch library; built into --output if absent")
    parser.add_argument("--check", action="store_true",
                        help="compare this run against the tracked results.json instead of writing it")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    _require_frozen_expectations()
    library = str(Path(args.library).resolve()) if args.library else _build_library(output)

    summary = run_study(output, library)
    _write_json(output / "results.json", summary)
    g3 = summary["cells"]["g3_gpu_side"]
    relations = summary["cells"]["g4_live"]["relations"]
    print(json.dumps({key: summary[key] for key in ("evidence", "findings", "status")},
                     indent=2, sort_keys=True))
    print("G3 held:", {name: row["held"] for name, row in g3.items()})
    print("G4 relations:", {name: row["held"] for name, row in relations.items()})
    print("G4 strict b200 > h100 cells:",
          len(relations["b200_at_or_above_h100_fan_in_and_live"]["strict_cells"]))
    if args.check:
        tracked = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if comparable(summary) != comparable(tracked):
            raise SystemExit("this run disagrees with the tracked results.json")
        print("check: this run reproduces the tracked results.json")
    else:
        _write_json(RESULTS_PATH, summary)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
