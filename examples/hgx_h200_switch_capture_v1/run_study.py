"""Run the frozen PLACE-12 HGX H200 switch capture qualification.

The study answers one question: can the `h100` generation of the eight-GPU
switched preset be bound, switch side included, to a real HGX H200 board (the
NVLink 4 baseboard of HGX H100), so that the declared per-chip bundle equals
the captured per-switch lane counts, every preset switch port lands on one
distinct captured switch port, every route path joins two ports of one
captured switch, and no request metric or accepted artifact moves?

The input is the rented board fixture under `tests/fixtures/nccl_topology/`,
whose container passes the four NVSwitches through. Cells H1 through H4 and H7
are structural exact guards, H6 is a rejection control family, and the three
preset digests, both switched studies' tracked results and their `--check`
runs are fatal by-construction identities. The expectations-only amendment
recorded in `AMENDMENT_COMMIT` adds the device-binding and switch-list controls
and the NIC facts of H2, and moves the B200 results identity to the file as
regenerated on the stacked B200 branch.

Cell H5 is live and scored. It runs the DGX study's live cells at generation
`h100` (tensor-parallel all-reduce and expert dispatch and combine at widths
2, 4 and 8, attachment rates 12.5 and 25 GB/s per lane, Python and native
switch selection) through that study's own `live_config` and `request`
helpers with the declared preset and with two captured fabrics substituted
into the same configuration: the full capture binding (module-id slots and
captured switch ports) and the switch ports alone. Packet observations are
compared after mapping every port, link and switch identity, and every
numeric port index, of both fabrics to its (rank, chip, lane) position. The
causal logs (packets, packet events, resource visits, physical paths, extents,
buffer claims and the drained result) are compared in order. Snapshots the
runtime orders by identity or by fabric element order (buffer ownership, and
for the slot-permuted binding the port, context and snapshot lists and the
binding inventory) are compared as sets, because renaming or reslotting a
GPU reorders them without changing any row.

Usage:

    python examples/hgx_h200_switch_capture_v1/run_study.py --output <dir> [--library <lib>]
    python examples/hgx_h200_switch_capture_v1/run_study.py --output <dir> --library <lib> --check
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
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTATIONS_COMMIT = "3d24b2b1d5a1300e9b7fdb2000acc5167eab94a9"
AMENDMENT_COMMIT = "e3a94f3490377f26f7ce397a32881d620055b0e2"
#: The stacked B200 branch commit whose regenerated results the amendment pins.
B200_BASE_COMMIT = "33dae4f636c1a12c824fe1ae45d815ac50d12f95"
RESULT_SCHEMA = "simllm-hgx-h200-switch-capture-result-v1"
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-13.json"
FROZEN_FILES = {
    EXPECTATIONS_COMMIT: ("expectations.md", "expectations.json"),
    AMENDMENT_COMMIT: ("expectations-amendment-2026-09-13.md",
                       "expectations-amendment-2026-09-13.json"),
}
RESULTS_PATH = STUDY_DIR / "results.json"
FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "nccl_topology" / "vastai_hgx_h200_8x"
DGX_STUDY = REPOSITORY_ROOT / "examples" / "dgx_nvlink_v1"
B200_STUDY = REPOSITORY_ROOT / "examples" / "hgx_b200_capture_v1"
REMOTE_BLOCK = "nvidia-smi nvlink -R (remote pci bus id per link)"
NVSWITCH_BLOCK = "pci devices class 0x0680 (nvswitch/bridge)"
MODULE_BLOCK = "nvidia-smi -q (GPU board/module ids)"
FULL_QUERY_BLOCK = "nvidia-smi -q full"
LSPCI_BLOCK = "lspci nvidia/bridges/nic"

NODE_ID = "hgx-h200"
LIVE_NODE_ID = "node-0"
PROPAGATION_DELAY_PS = 1000
SWITCH_INPUT_BUFFER_BYTES = 65536
#: The DGX study's frozen live grid, restated because its `run` builds it inline.
RATES = (12_500_000_000, 25_000_000_000)
WIDTHS = (2, 4, 8)
KINDS = ("tp", "ep-balanced", "ep-skewed")
#: The B200 preset digest its capture tests pin, a post-implementation value.
B200_PRESET_DIGEST = {"bytes": 125_741,
                      "sha256": "ef920aa9abc7b61222880d97b60d2e016b37ed1df46b5c628488054553f5163f"}
#: Declared NVLink 4 payload bound of the physical-sanity section: 18 lanes at 25 GB/s.
H100_GPU_CEILING_BYTES_PER_SECOND = 18 * 25_000_000_000


def _encode(value):
    if isinstance(value, Fraction):
        return {"denominator": value.denominator, "numerator": value.numerator}
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=_encode) + "\n").encode()


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, check=True,
                          capture_output=True).stdout


def _require_frozen_expectations() -> None:
    """Refuse to run unless both freeze commits precede HEAD and are unchanged."""

    for commit in (*FROZEN_FILES, B200_BASE_COMMIT):
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                  cwd=REPOSITORY_ROOT, check=False)
        if ancestor.returncode != 0:
            raise SystemExit(f"commit {commit} is not an ancestor of HEAD")
    for commit, names in FROZEN_FILES.items():
        for name in names:
            relative = (STUDY_DIR / name).relative_to(REPOSITORY_ROOT).as_posix()
            if (STUDY_DIR / name).read_bytes() != _git("show", f"{commit}:{relative}"):
                raise SystemExit(f"{name} differs from frozen commit {commit}")


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes().replace(b"\r\n", b"\n")


def _inventory() -> dict[str, tuple[str, ...]]:
    from simllm.placement import read_inventory_sections

    return read_inventory_sections(_fixture_bytes("node_inventory.txt").decode("utf-8"))


def _switch_table(inventory: dict[str, tuple[str, ...]], dump, remote=None):
    from simllm.placement import (
        captured_switch_ports,
        read_gpu_bus_ids,
        read_gpu_minor_uuids,
        read_nvlink_remote_ports,
        read_nvswitch_list,
    )

    return captured_switch_ports(
        read_nvlink_remote_ports(inventory[REMOTE_BLOCK]) if remote is None else remote,
        read_nvswitch_list(inventory[NVSWITCH_BLOCK]),
        bus_id_by_uuid=read_gpu_bus_ids(inventory[FULL_QUERY_BLOCK]),
        uuid_by_minor=read_gpu_minor_uuids(inventory[FULL_QUERY_BLOCK]),
        bus_id_by_gpu_dev={gpu.dev: gpu.busid for gpu in dump.gpus},
    )


def _query_records(lines) -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    for line in lines:
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9 ]*?)\s*:\s*(.*?)\s*", line)
        if match is None:
            continue
        key, value = match.groups()
        if key == "Product Name":
            records.append({})
        if records:
            records[-1].setdefault(key, []).append(value)
    return records


def _join(dump, inventory, **overrides):
    from simllm.placement import captured_switched_node, read_module_ids

    arguments = {
        "generation": "h100",
        "node_id": NODE_ID,
        "pool_role": "serving",
        "global_rank_by_gpu_dev": {dev: dev for dev in range(8)},
        "propagation_delay_ps": PROPAGATION_DELAY_PS,
        "switch_input_buffer_bytes": SWITCH_INPUT_BUFFER_BYTES,
        "module_id_by_gpu_dev": read_module_ids(inventory[MODULE_BLOCK]),
        "switch_ports_by_gpu_dev": _switch_table(inventory, dump),
    }
    arguments.update(overrides)
    return captured_switched_node(dump, **arguments)


def _captured_pair(port_id: str) -> tuple[str, int]:
    match = re.search(r":switch-([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]):port-([0-9]+)$",
                      port_id)
    if match is None:
        raise ValueError(f"{port_id} is not a captured switch port identity")
    return match.group(1), int(match.group(2))


def run_h1(dump) -> dict[str, Any]:
    """Cell H1: the parse of the switched dump."""

    switches = [device for device in dump.pci if device.pci_class == "0x060400"]
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    devices = {device.busid: device for device in dump.pci}
    graphs = {graph.get("pattern"): graph
              for graph in ElementTree.fromstring(_fixture_bytes("nccl_graph.xml")).iter("graph")}
    return {
        "counts": {
            "counts_in_busid_order": sorted({
                tuple(row.count for row in sorted(dump.switch_rows(gpu.busid),
                                                   key=lambda row: row.target))
                for gpu in gpus
            }),
            "cpus": len(dump.cpus),
            "distinct_socket_nics": len(dump.distinct_nets),
            "gpus": len(gpus),
            "nic_repetitions": len(dump.nets),
            "pcie_switches": len(switches),
            "switch_attached_rows": sum(row.switch_attached for row in dump.nvlinks),
        },
        "gpu": {
            "busid_by_dev": {str(gpu.dev): gpu.busid for gpu in gpus},
            "device": sorted({devices[gpu.busid].device for gpu in gpus}),
            "gdr": sorted({gpu.gdr for gpu in gpus}),
            "numa_by_dev": {str(gpu.dev): dump.numa_of(gpu.busid) for gpu in gpus},
            "pcie_switch_by_dev": {str(gpu.dev): devices[gpu.busid].parent_busid for gpu in gpus},
            "sm": sorted({gpu.sm for gpu in gpus}),
            "vendor": sorted({devices[gpu.busid].vendor for gpu in gpus}),
        },
        "pcie_switch": {"device": sorted({device.device for device in switches}),
                        "vendor": sorted({device.vendor for device in switches})},
        "switch_rows": {
            "dev0_dump_order": [row.target for row in dump.switch_rows(gpus[0].busid)],
            "targets": sorted({row.target for row in dump.nvlinks}),
            "tclass": sorted({row.tclass for row in dump.nvlinks}),
        },
        "nccl": {
            "env": _fixture_bytes("nccl_env.txt").decode("utf-8").splitlines()[0],
            # NCCL graph patterns: 4 is ring, 1 is balanced tree, 5 is NVLS.
            "nvls_channels": int(graphs["5"].get("nchannels")),
            "ring_channels": int(graphs["4"].get("nchannels")),
            "tree_channels": int(graphs["1"].get("nchannels")),
        },
        "socket_nic": [{"gdr": net.gdr, "name": net.name, "speed_mbit": net.speed}
                       for net in dump.distinct_nets],
    }


def run_h2(dump, inventory) -> dict[str, Any]:
    """Cell H2: the inventory readers and the NIC facts beside them."""

    from simllm.placement import (
        read_gpu_bus_ids,
        read_module_ids,
        read_nvlink_remote_ports,
        read_nvswitch_list,
    )

    blocks = read_nvlink_remote_ports(inventory[REMOTE_BLOCK])
    remote = {gpu: block.links for gpu, block in blocks.items()}
    bus_ids = read_gpu_bus_ids(inventory[FULL_QUERY_BLOCK])
    devices = read_nvswitch_list(inventory[NVSWITCH_BLOCK])
    functions = [line.split(" ", 1)[0] for line in inventory[LSPCI_BLOCK]
                 if "Mellanox Technologies ConnectX" in line and "Virtual Function" in line]
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    matrix = [ansi.sub("", line) for line in inventory["nvidia-smi topo -m"]]
    header = matrix[0].split("\t")
    nics = [index for index, name in enumerate(header) if re.fullmatch(r"NIC[0-9]+", name)]
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in matrix if re.match(r"GPU[0-9]+\t", line)}
    modules = read_module_ids(inventory[MODULE_BLOCK])
    pairs = [pair for rows in remote.values() for pair in rows]
    switches = sorted({busid for busid, _ in pairs})
    per_switch = Counter(busid for busid, _ in set(pairs))
    return {
        "connectx_virtual_functions": functions,
        "pix_nics_by_gpu": [[header[index] for index in nics if rows[dev][index].strip() == "PIX"]
                            for dev in sorted(rows)],
        "remote_blocks_head_dump_devices": all(
            bus_ids.get(block.uuid) == next(g.busid for g in dump.gpus if g.dev == gpu)
            for gpu, block in blocks.items()),
        "distinct_pairs": len(set(pairs)),
        "module_id_by_dev": {str(dev): module for dev, module in sorted(modules.items())},
        "per_gpu": sorted({len(rows) for rows in remote.values()}),
        "per_switch_counts": sorted({
            tuple(Counter(busid for busid, _ in rows)[switch] for switch in switches)
            for rows in remote.values()
        }),
        "ports_per_switch": [per_switch[switch] for switch in switches],
        "remote_rows": len(pairs),
        "slots": [modules[dev] - 1 for dev in range(8)],
        "switch_devices": [asdict(device) for device in devices],
    }


def run_h3(dump, inventory) -> dict[str, Any]:
    """Cell H3: the switch-side binding to the h100 preset."""

    from simllm.placement.dgx import DGX_NVLINK_BUNDLES

    table = _switch_table(inventory, dump)
    bundle = DGX_NVLINK_BUNDLES["h100"]
    switches = sorted({busid for rows in table.values() for busid, _ in rows})
    counts = {dev: tuple(Counter(busid for busid, _ in rows)[switch] for switch in switches)
              for dev, rows in table.items()}
    _, bound = _join(dump, inventory)
    _, declared = _join(dump, inventory, switch_ports_by_gpu_dev=None)
    switch_ports = [port for port in bound.ports if port.switch_id is not None]
    bound_pairs = [_captured_pair(port.port_id) for port in switch_ports]
    captured_pairs = {pair for rows in table.values() for pair in rows}
    two_ports_one_switch = True
    chip_equals_switch = True
    for source, destination in permutations(range(8), 2):
        for bound_path, declared_path in zip(bound.paths_between(source, destination),
                                             declared.paths_between(source, destination),
                                             strict=True):
            two_ports_one_switch &= (
                bound_path.switch_input_port_id.startswith(bound_path.switch_id + ":port-")
                and bound_path.switch_output_port_id.startswith(bound_path.switch_id + ":port-")
                and bound_path.input_link.link_id == declared_path.input_link.link_id
                and bound_path.output_link.link_id == declared_path.output_link.link_id
            )
            chip = int(declared_path.switch_id.rsplit("switch-", 1)[1])
            chip_equals_switch &= bound_path.switch_id.endswith(f":switch-{switches[chip - 1]}")
    per_switch_links = Counter(_captured_pair(link.endpoint_b)[0] for link in bound.links)
    return {
        "bundle_equals_captured_counts": {
            "bundle": list(bundle),
            "captured_counts": sorted({list(value) == list(bundle) for value in counts.values()}),
            "held": all(value == bundle for value in counts.values()),
        },
        "switch_ports_bijective": {
            "bound_switch_ports": len(switch_ports),
            "captured_pairs": len(captured_pairs),
            "held": len(switch_ports) == len(set(bound_pairs)) == len(captured_pairs) == 144
            and set(bound_pairs) == captured_pairs,
        },
        "every_path_two_ports_one_switch": {"held": two_ports_one_switch,
                                            "ordered_pairs": 56, "paths_per_pair": 18},
        "chip_index_equals_switch": {"held": chip_equals_switch, "switches_in_chip_order": switches},
        "port_totals_conserve": {
            "held": [per_switch_links[switch] for switch in switches] == [32, 40, 40, 32],
            "links_per_switch": [per_switch_links[switch] for switch in switches],
        },
        "h100_bundle_confirmed": {
            "held": all(value == bundle for value in counts.values()) and len(switches) == len(bundle),
            "switches": len(switches),
        },
    }


def run_h4(dump, inventory) -> dict[str, Any]:
    """Cell H4: the GPU side of the h100 binding against nvidia-smi."""

    node, bound = _join(dump, inventory)
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    status: dict[int, list[str]] = {}
    for line in inventory["nvidia-smi nvlink -s"]:
        heading = re.match(r"GPU ([0-9]+): ", line)
        if heading:
            status[int(heading.group(1))] = []
        elif line.strip() and status:
            link = re.fullmatch(r"\s*Link [0-9]+: (.+)", line)
            status[max(status)].append(link.group(1) if link else line)
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    matrix = [ansi.sub("", line) for line in inventory["nvidia-smi topo -m"]]
    numa_column = matrix[0].split("\t").index("NUMA Affinity")
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in matrix if re.match(r"GPU[0-9]+\t", line)}
    listed = [re.fullmatch(r"GPU ([0-9]+): NVIDIA H200 \(UUID: (GPU-[0-9a-f-]+)\)", line)
              for line in inventory["nvidia-smi -L"]]
    full = _query_records(inventory["nvidia-smi -q full"])
    board = _query_records(inventory[MODULE_BLOCK])
    modules = {int(record["Minor Number"][0]): int(record["Module Id"][0]) for record in board}
    parents = [next(d.parent_busid for d in dump.pci if d.busid == gpu.busid) for gpu in gpus]
    slot_ports = all(f":gpu-{modules[port.gpu_rank] - 1}:" in port.port_id
                     for port in bound.ports if port.gpu_rank is not None)
    owner = {port.port_id: port.gpu_rank for port in bound.ports}
    slot_links = all(
        f":link-{modules[owner[link.endpoint_a]] - 1}-" in link.link_id for link in bound.links)
    return {
        "lanes_equal_links": {
            "held": status == {dev: [status[dev][0]] * 18 for dev in range(8)}
            and len(status) == 8,
            "signalling_rates": sorted({rate for rates in status.values() for rate in rates}),
        },
        "nv18_equals_paths_and_dump_total": {
            "held": sorted(rows) == list(range(8)) and all(
                rows[a][1 + b].strip() == "NV18" and len(bound.paths_between(a, b)) == 18
                for a, b in permutations(range(8), 2))
            and all(dump.switch_lanes(gpu.busid) == 18 for gpu in gpus),
        },
        "order_busids_pcie_numa": {
            "held": [match is not None and int(match.group(1)) for match in listed]
            == list(range(8))
            and [record["GPU UUID"][0] for record in full] == [m.group(2) for m in listed]
            and [record["Bus Id"][0].lower().replace("00000000:", "0000:") for record in full]
            == [gpu.gpu_id for gpu in node.gpus] == [gpu.busid for gpu in gpus]
            and len(set(parents)) == 8
            and [int(rows[dev][numa_column]) for dev in range(8)]
            == [dump.numa_of(gpu.busid) for gpu in gpus]
            and Counter(dump.numa_of(gpu.busid) for gpu in gpus) == {0: 4, 1: 4},
            "pcie_locations": [gpu.pcie_location for gpu in node.gpus],
        },
        "distinct_board_ids_one_part_number": {
            "board_ids": [record["Board ID"][0] for record in board],
            "held": len({record["Board ID"][0] for record in board}) == 8
            and len({part for record in full for part in record.get("Board Part Number", [])}) == 1,
            "part_numbers": sorted({part for record in full
                                    for part in record.get("Board Part Number", [])}),
        },
        "module_id_slots": {
            "held": slot_ports and slot_links,
            "slot_by_dev": {str(dev): modules[dev] - 1 for dev in range(8)},
        },
    }


def _identity_names(fabric) -> dict[str, str]:
    """Map a preset-shaped domain's identities to (rank, chip, lane) positions."""

    from simllm.placement.dgx import DGX_NVLINK_BUNDLES

    names: dict[str, str] = {}
    index = 0
    for chip, width in enumerate(DGX_NVLINK_BUNDLES["h100"], 1):
        for _slot in range(8):
            for lane in range(width):
                link = fabric.links[index]
                gpu_port, switch_port = fabric.ports[2 * index], fabric.ports[2 * index + 1]
                if (link.endpoint_a, link.endpoint_b) != (gpu_port.port_id, switch_port.port_id):
                    raise ValueError("peer fabric is not in preset element order")
                key = f"r{gpu_port.gpu_rank}c{chip}l{lane}"
                names[link.link_id] = f"LINK[{key}]"
                names[gpu_port.port_id] = f"GPU-PORT[{key}]"
                names[switch_port.port_id] = f"SWITCH-PORT[{key}]"
                names[switch_port.switch_id] = f"CHIP[{chip}]"
                index += 1
    return names


def _canonical_item(item) -> str:
    return json.dumps(item, sort_keys=True)


def _sorted_inventory(value):
    """Sort every list of a static inventory, innermost first."""

    if isinstance(value, dict):
        return {key: _sorted_inventory(item) for key, item in value.items()}
    if isinstance(value, list):
        return sorted((_sorted_inventory(item) for item in value), key=_canonical_item)
    return value


def _port_indices(value, ports: dict[int, str]):
    if isinstance(value, dict):
        return {key: (ports[item] if key in ("input_port", "output_port") and type(item) is int
                      else _port_indices(item, ports)) for key, item in value.items()}
    if isinstance(value, list):
        return [_port_indices(item, ports) for item in value]
    return value


def _positional_observations(observations, fabric, *, reslotted: bool) -> str:
    names = _identity_names(fabric)
    ports = {index: names[port.port_id] for index, port in enumerate(fabric.ports)}
    pattern = re.compile("|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)))
    text = json.dumps(observations, sort_keys=True, default=_encode)
    rows = _port_indices(json.loads(pattern.sub(lambda match: names[match.group(0)], text)), ports)
    for row in rows:
        row["buffer_ownership"] = sorted(row["buffer_ownership"], key=_canonical_item)
        if reslotted:
            for key in ("contexts", "ports", "port_snapshots"):
                row[key] = sorted(row[key], key=_canonical_item)
            row["binding"] = _sorted_inventory(row["binding"])
    return json.dumps(rows, sort_keys=True)


def run_h5(dump, inventory, output: Path, library: str) -> dict[str, Any]:
    """Cell H5: captured switch-bound and declared fabrics give the same run."""

    from examples.dgx_nvlink_v1 import run_study as dgx
    from simllm.backends.peer_step import PeerPacketConfig
    from simllm.placement import FabricTopologyManifest

    captured = {
        "bound": _join(dump, inventory, node_id=LIVE_NODE_ID)[1],
        "switch-only": _join(dump, inventory, node_id=LIVE_NODE_ID, module_id_by_gpu_dev=None)[1],
    }
    raw = output / "h5"
    raw.mkdir(parents=True, exist_ok=True)

    def metrics(result):
        return dgx.canonical({key: result[key] for key in ("jct_ps", "steps", "tpot_ps", "ttft_ps")})

    cells = []
    for rate, width, kind in product(RATES, WIDTHS, KINDS):
        runs, fabrics = {}, {}
        for arm, selected in (("python", None), ("native", library)):
            config = dgx.live_config(raw / f"{rate}-{width}-{kind}-declared-{arm}", "h100", rate,
                                     width, kind, selected)
            fabrics["declared"] = config.peer_packet.fabric.peer_fabrics[0]
            runs["declared", arm] = dgx.request(config)
            for label, fabric in captured.items():
                base = dgx.live_config(raw / f"{rate}-{width}-{kind}-{label}-{arm}", "h100", rate,
                                       width, kind, selected)
                rated = replace(fabric, links=tuple(replace(link, link_rate_bps=8 * rate)
                                                    for link in fabric.links))
                fabrics[label] = rated
                topology = FabricTopologyManifest(nodes=list(base.peer_packet.fabric.nodes),
                                                  peer_fabrics=(rated,))
                runs[label, arm] = dgx.request(replace(base, peer_packet=PeerPacketConfig(
                    topology, base.peer_packet.profiles, native_switch_library=selected)))
        (raw / f"{rate}-{width}-{kind}-bound-native.json").write_bytes(
            dgx.canonical(runs["bound", "native"]))

        def observations(label, arm, reslotted, runs=runs, fabrics=fabrics):
            return _positional_observations(dgx.comparable(runs[label, arm])["observations"],
                                            fabrics[label], reslotted=reslotted)

        cells.append({
            "jct_ps": runs["bound", "native"]["jct_ps"],
            "kind": kind,
            "packet_observations_equal": all(
                observations("bound", arm, True) == observations("declared", arm, True)
                for arm in ("python", "native")),
            "python_native_exact": all(
                dgx.canonical(dgx.comparable(runs[label, "python"]))
                == dgx.canonical(dgx.comparable(runs[label, "native"]))
                for label in ("declared", "bound", "switch-only")),
            "rate": rate,
            "request_metrics_equal": all(
                metrics(runs[label, arm]) == metrics(runs["declared", arm])
                for label in ("bound", "switch-only") for arm in ("python", "native")),
            "step_latency_ps": [step["result"]["step_latency_ps"]
                                for step in runs["bound", "native"]["steps"]],
            "switch_only_observations_equal": all(
                observations("switch-only", arm, False) == observations("declared", arm, False)
                for arm in ("python", "native")),
            "width": width,
        })
    return {
        "cells": cells,
        "endpoint_ceiling_bytes_per_second": 450 * 10**9,
        "held": all(cell[key] for cell in cells for key in (
            "packet_observations_equal", "python_native_exact", "request_metrics_equal",
            "switch_only_observations_equal")),
        "live_cells": len(cells),
    }


def _h6_cases(dump, inventory):
    from simllm.placement import NcclTopologyDump, read_nvlink_remote_ports, read_nvswitch_list

    table = _switch_table(inventory, dump)

    def five_on_c3():
        rows = list(table[0])
        index = next(i for i, (busid, _) in enumerate(rows) if busid == "0000:c6:00.0")
        rows[index] = ("0000:c3:00.0", 99)
        return _join(dump, inventory, switch_ports_by_gpu_dev={**table, 0: tuple(rows)})

    def port_twice():
        shared = next(pair for pair in table[0] if pair[0] == "0000:c3:00.0")
        rows = list(table[1])
        index = next(i for i, (busid, _) in enumerate(rows) if busid == "0000:c3:00.0")
        rows[index] = shared
        return _join(dump, inventory, switch_ports_by_gpu_dev={**table, 1: tuple(rows)})

    def virtual_row():
        root = ElementTree.fromstring(_fixture_bytes("nccl_topo.xml"))
        root.find(".//gpu[@dev='0']").append(ElementTree.Element(
            "nvlink", {"target": "fffffff:ff:ff.0", "count": "18", "tclass": "0x068000"}))
        return _join(NcclTopologyDump.parse(ElementTree.tostring(root, encoding="unicode")),
                     inventory)

    def three_switches():
        edited = dict(inventory)
        edited[NVSWITCH_BLOCK] = tuple(line for line in inventory[NVSWITCH_BLOCK]
                                       if not line.startswith("0000:c6:00.0"))
        return _join(dump, inventory, switch_ports_by_gpu_dev=_switch_table(edited, dump))

    def seventeen_links():
        edited = dict(inventory)
        lines = list(inventory[REMOTE_BLOCK])
        heading = next(i for i, line in enumerate(lines) if line.startswith("GPU 3:"))
        del lines[heading + 18]
        edited[REMOTE_BLOCK] = tuple(lines)
        return _join(dump, inventory, switch_ports_by_gpu_dev=_switch_table(edited, dump))

    def permuted(change):
        blocks = read_nvlink_remote_ports(inventory[REMOTE_BLOCK])
        return lambda: _switch_table(inventory, dump, change(blocks))

    def swapped_uuids(blocks):
        return {**blocks, 0: replace(blocks[0], uuid=blocks[1].uuid),
                1: replace(blocks[1], uuid=blocks[0].uuid)}

    extra_switch = "0000:c7:00.0 class=0x068000 vendor=0x10de device=0x22a3 numa=1"

    def unused_switch():
        edited = dict(inventory)
        edited[NVSWITCH_BLOCK] = (*inventory[NVSWITCH_BLOCK], extra_switch)
        return _switch_table(edited, dump)

    def absent_switch():
        root = ElementTree.fromstring(_fixture_bytes("nccl_topo.xml"))
        root.find(".//gpu[@dev='0']/nvlink[@target='0000:c3:00.0']").set("target", "0000:c7:00.0")
        return _join(NcclTopologyDump.parse(ElementTree.tostring(root, encoding="unicode")),
                     inventory)

    misplaced = "but the NCCL dump places dev 0 at 0000:83:00.0"
    return {
        "swapped gpus": (permuted(lambda blocks: {**blocks, 0: blocks[1], 1: blocks[0]}),
                         misplaced),
        "rotated gpus": (permuted(lambda blocks: {gpu: blocks[(gpu + 1) % 8] for gpu in blocks}),
                         misplaced),
        "swapped uuids": (permuted(swapped_uuids), misplaced),
        "switch list class or vendor": (
            lambda: read_nvswitch_list([*inventory[NVSWITCH_BLOCK],
                                        extra_switch.replace("vendor=0x10de", "vendor=0x1000")]),
            "listed switch 0000:c7:00.0 has vendor 0x1000, not 0x10de"),
        "unused listed switch": (unused_switch, "listed NVSwitch 0000:c7:00.0 receives no lane"),
        "dump switch absent from list": (
            absent_switch, ("NCCL switch row of GPU dev 0 names 0000:c7:00.0, which is absent "
                            "from the captured switch list")),
        "5 lanes on c3": (five_on_c3, ("GPU dev 0 has 5 captured lanes on switch 0000:c3:00.0; "
                                       "chip 1 of generation h100 requires 4")),
        "switch port used twice": (port_twice, "is used by two links"),
        "virtual row beside real rows": (
            virtual_row, "gpu 0000:83:00.0 mixes the virtual NVSwitch row with real switch rows"),
        "three switches": (three_switches,
                           "GPU 0 link 12 reaches 0000:c6:00.0, which is not a listed NVSwitch"),
        "b200 against four switches": (
            lambda: _join(dump, inventory, generation="b200"),
            "generation b200 binds 2 NVSwitch chips; the captured links name 4 switches"),
        "17 links for one gpu": (
            seventeen_links, "GPU dev 3 has 17 captured NVLink links; generation h100 requires 18"),
    }


def run_h6(dump, inventory) -> dict[str, Any]:
    """Cell H6: each refusal, raised before any schema object exists."""

    from simllm.placement import nccl_topology

    names = ("FabricNodePlacement", "GpuFabricPlacement", "dgx_peer_fabric")
    originals = {name: getattr(nccl_topology, name) for name in names}
    cases = _h6_cases(dump, inventory)
    built: list[str] = []
    rows: dict[str, Any] = {}

    def sentinel(name):
        def build(*_args, **_kwargs):
            built.append(name)
            raise AssertionError(f"{name} built before the refusal")
        return build

    try:
        for name in names:
            setattr(nccl_topology, name, sentinel(name))
        for label, (build, expected) in cases.items():
            built.clear()
            try:
                build()
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


def run_h7(dump, inventory, output: Path, library: str) -> dict[str, Any]:
    """Cell H7: wire identity, accepted digests and both switched studies' checks."""

    from simllm.placement import FabricTopologyManifest, dgx_peer_fabric

    node, bound = _join(dump, inventory)
    fabric = FabricTopologyManifest(nodes=[node], peer_fabrics=(bound,), source="extracted")
    cell = output / "h7"
    cell.mkdir(parents=True, exist_ok=True)
    first = fabric.save(cell / "fabric.json")
    loaded = FabricTopologyManifest.load(first)
    second = loaded.save(cell / "fabric-round-trip.json")
    digests = {}
    for generation in ("a100", "h100", "b200"):
        payload = json.dumps(asdict(dgx_peer_fabric(
            generation, node_id="node-0", ranks=tuple(range(8)), propagation_delay_ps=1000,
            switch_input_buffer_bytes=65536)), sort_keys=True, separators=(",", ":")).encode()
        digests[generation] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    checks = {}
    for label, script, arguments in (
        ("dgx", DGX_STUDY / "run_study.py", ["--library", library, "--output",
                                             str(output / "dgx-check"), "--check"]),
        ("b200", B200_STUDY / "run_study.py", ["--library", library, "--output",
                                               str(output / "b200-check"), "--check"]),
    ):
        completed = subprocess.run([sys.executable, str(script), *arguments], cwd=REPOSITORY_ROOT,
                                   capture_output=True, text=True, check=False)
        (cell / f"{label}-check.log").write_text(completed.stdout + completed.stderr,
                                                 encoding="utf-8")
        checks[label] = completed.returncode == 0
    b200_relative = (B200_STUDY / "results.json").relative_to(REPOSITORY_ROOT).as_posix()
    return {
        "b200_results_sha256": hashlib.sha256((B200_STUDY / "results.json").read_bytes()).hexdigest(),
        "b200_results_sha256_at_stacked_base": hashlib.sha256(
            _git("show", f"{B200_BASE_COMMIT}:{b200_relative}")).hexdigest(),
        "dgx_results_sha256": hashlib.sha256((DGX_STUDY / "results.json").read_bytes()).hexdigest(),
        "preset_digests": digests,
        "round_trip": loaded == fabric and first.read_bytes() == second.read_bytes(),
        "study_checks": checks,
    }


def analyze(cells: dict[str, Any], frozen: dict[str, Any], amendment: dict[str, Any],
            fixture_sha256: dict[str, str]) -> dict[str, Any]:
    """Apply every frozen guard, then score the H5 identity family."""

    fatal: list[str] = []
    structural: list[str] = []
    rejection: list[str] = []
    scored: list[str] = []
    capture = frozen["capture"]
    for name, digest in frozen["baseline"]["fixture"].items():
        if name != "directory" and fixture_sha256.get(name) != digest:
            fatal.append(f"fixture digest {name}")

    h1, expected = cells["h1_parse"], frozen["cells"]["h1_parse"]
    counts = h1["counts"]
    if (counts["cpus"], counts["pcie_switches"], counts["gpus"], counts["switch_attached_rows"],
            counts["counts_in_busid_order"], counts["distinct_socket_nics"],
            counts["nic_repetitions"]) != (
            expected["cpus"], expected["pcie_switches"], expected["gpus"],
            expected["switch_attached_rows"], [expected["counts_in_busid_order"]],
            expected["distinct_socket_nics"], expected["nic_repetitions"]):
        structural.append("h1: element counts")
    gpu = capture["gpu"]
    if (h1["gpu"]["busid_by_dev"], h1["gpu"]["numa_by_dev"], h1["gpu"]["pcie_switch_by_dev"],
            h1["gpu"]["vendor"], h1["gpu"]["device"], h1["gpu"]["sm"], h1["gpu"]["gdr"]) != (
            gpu["busid_by_dev"], gpu["numa_by_dev"], gpu["pcie_switch_by_dev"], [gpu["vendor"]],
            [gpu["device"]], [gpu["sm"]], [1]):
        structural.append("h1: GPU literals")
    if h1["pcie_switch"] != {"device": ["0x8232"], "vendor": ["0x104c"]}:
        structural.append("h1: PCIe switch literals")
    if (h1["switch_rows"]["targets"], h1["switch_rows"]["tclass"]) != (
            capture["nvswitch"]["busids"], [capture["nvswitch"]["class"]]):
        structural.append("h1: switch rows")
    nccl = capture["nccl"]
    if (h1["nccl"]["ring_channels"], h1["nccl"]["tree_channels"], h1["nccl"]["nvls_channels"]) != (
            nccl["ring_channels"], nccl["tree_channels"], nccl["nvls_channels"]) or (
            f"nccl ({nccl['version'].replace('.', ', ')})" not in h1["nccl"]["env"]):
        structural.append("h1: NCCL version or graph channels")
    if h1["socket_nic"] != [{"gdr": 0, "name": capture["nic"]["socket"],
                             "speed_mbit": capture["nic"]["socket_speed_mbit"]}]:
        structural.append("h1: socket NIC")

    h2, expected = cells["h2_inventory"], frozen["cells"]["h2_inventory"]
    if (h2["remote_rows"], h2["per_gpu"], h2["per_switch_counts"], h2["distinct_pairs"],
            h2["ports_per_switch"], len(h2["switch_devices"]), h2["slots"]) != (
            expected["remote_rows"], [expected["per_gpu"]], [expected["per_switch_counts"]],
            expected["distinct_pairs"], expected["ports_per_switch"], expected["switch_devices"],
            expected["slots"]):
        structural.append("h2: inventory tables")
    if h2["module_id_by_dev"] != gpu["module_id_by_dev"]:
        structural.append("h2: module ids")
    if (len(h2["connectx_virtual_functions"]) != capture["nic"]["connectx_virtual_functions"]
            or h2["pix_nics_by_gpu"] != [[f"NIC{dev}"] for dev in range(8)]):
        structural.append("h2: ConnectX virtual functions and PIX affinity")
    if h2["remote_blocks_head_dump_devices"] is not True:
        structural.append("h2: nvlink -R blocks do not head the dump devices")
    if [(row["busid"], row["vendor"], row["device"], row["numa"]) for row in h2["switch_devices"]] != [
            (busid, capture["nvswitch"]["vendor"], capture["nvswitch"]["device"],
             capture["nvswitch"]["numa"]) for busid in capture["nvswitch"]["busids"]]:
        structural.append("h2: NVSwitch devices")

    for cell, names in (("h3_switch_side", frozen["cells"]["h3_switch_side"]),
                        ("h4_gpu_side", frozen["cells"]["h4_gpu_side"])):
        for name in names:
            if cells[cell].get(name, {}).get("held") is not True:
                structural.append(f"{cell[:2]}: {name}")
    if cells["h4_gpu_side"]["distinct_board_ids_one_part_number"]["board_ids"] != gpu["board_ids"]:
        structural.append("h4: board ids")
    if cells["h4_gpu_side"]["lanes_equal_links"]["signalling_rates"] != [
            f"{capture['nvlink']['signalling_gb_per_s']} GB/s"]:
        structural.append("h4: signalling rate")

    h5 = cells["h5_live_identity"]
    if h5["endpoint_ceiling_bytes_per_second"] > H100_GPU_CEILING_BYTES_PER_SECOND:
        fatal.append("h5: endpoint ceiling above the 450 GB/s NVLink 4 bound")
    if h5["live_cells"] != len(RATES) * len(WIDTHS) * len(KINDS):
        fatal.append("h5: live grid")
    if h5["held"] is not True:
        scored.append("h5: bound and declared runs differ")

    h6 = cells["h6_refusals"]
    h6_labels = [*frozen["cells"]["h6_refusals"], *amendment["added_controls"]]
    for label in h6_labels:
        if h6.get(label, {}).get("refused") is not True:
            rejection.append(f"h6: {label} was not refused before any schema object")

    h7 = cells["h7_off_path"]
    artifacts = frozen["baseline"]["artifacts"]
    for generation in ("a100", "h100"):
        if h7["preset_digests"][generation]["sha256"] != artifacts[f"dgx_{generation}_peer_fabric_json"]:
            fatal.append(f"h7: {generation} preset digest")
    if h7["preset_digests"]["b200"] != B200_PRESET_DIGEST:
        fatal.append("h7: b200 preset digest")
    if h7["dgx_results_sha256"] != artifacts["dgx_nvlink_v1_results"]:
        fatal.append("h7: DGX study results")
    if h7["b200_results_sha256"] != h7["b200_results_sha256_at_stacked_base"]:
        fatal.append("h7: B200 study results")
    for label, held in h7["study_checks"].items():
        if held is not True:
            fatal.append(f"h7: {label} study check")
    if h7["round_trip"] is not True:
        structural.append("h7: round trip")

    findings = [*fatal, *structural, *rejection, *scored]
    status = "VOID" if (fatal or structural or rejection) else ("FAIL" if scored else "PASS")
    return {
        "evidence": {
            "exact_oracle_family": {"held": h5["held"], "instances": h5["live_cells"]},
            "fatal_compatibility_identities": frozen["evidence"]["fatal_compatibility_identities"],
            "rejection_control_family": {
                "controls": len(h6_labels),
                "refused": sum(h6.get(label, {}).get("refused") is True for label in h6_labels),
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
    from simllm.placement import NcclTopologyDump

    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    dump = NcclTopologyDump.load(FIXTURE_DIR / "nccl_topo.xml")
    inventory = _inventory()
    cells = {
        "h1_parse": run_h1(dump),
        "h2_inventory": run_h2(dump, inventory),
        "h3_switch_side": run_h3(dump, inventory),
        "h4_gpu_side": run_h4(dump, inventory),
        "h5_live_identity": run_h5(dump, inventory, output, library),
        "h6_refusals": run_h6(dump, inventory),
        "h7_off_path": run_h7(dump, inventory, output, library),
    }
    cells = json.loads(_json_bytes(cells))
    fixture_sha256 = {name: hashlib.sha256(_fixture_bytes(name)).hexdigest()
                      for name in sorted(frozen["baseline"]["fixture"]) if name != "directory"}
    analysis = analyze(cells, frozen, amendment, fixture_sha256)
    summary = {
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
    parser = argparse.ArgumentParser(description="Run the PLACE-12 HGX H200 switch capture study")
    parser.add_argument("--output", type=Path, required=True,
                        help="external evidence directory, absent or empty")
    parser.add_argument("--library",
                        help="built native switch library; built into --output if absent")
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
    (output / "results.json").write_bytes(_json_bytes(summary))
    print(json.dumps({key: summary[key] for key in ("evidence", "findings", "status")},
                     indent=2, sort_keys=True))
    print("H3 held:", {name: row["held"] for name, row in summary["cells"]["h3_switch_side"].items()})
    print("H4 held:", {name: row["held"] for name, row in summary["cells"]["h4_gpu_side"].items()})
    print("H5 held:", summary["cells"]["h5_live_identity"]["held"],
          "cells:", summary["cells"]["h5_live_identity"]["live_cells"])
    print("H7 study checks:", summary["cells"]["h7_off_path"]["study_checks"])
    if args.check:
        tracked = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if comparable(summary) != comparable(tracked):
            raise SystemExit("this run disagrees with the tracked results.json")
        print("check: this run reproduces the tracked results.json")
    else:
        RESULTS_PATH.write_bytes(_json_bytes(summary))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
