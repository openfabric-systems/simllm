"""NVSwitch port binding of the H100 switched preset to a captured HGX H200 board.

Cells H1, H2, H3, H4, H6 and H7 of examples/hgx_h200_switch_capture_v1/
expectations.md, plus freeze pins on its JSON registry. Cell H5 (the live
identity) and both switched studies' check runs execute in the study harness.

Two digests are pinned below from tracked files rather than frozen literals:
the B200 preset digest the B200 capture tests pin (a post-implementation
value of that slice) and the B200 study results tracked at the base commit
af5c0110.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
from xml.etree import ElementTree

import pytest

from simllm.placement import (
    FabricTopologyManifest,
    NcclTopologyDump,
    captured_switch_ports,
    captured_switched_node,
    dgx_peer_fabric,
    nccl_topology,
    read_inventory_sections,
    read_module_ids,
    read_nvlink_remote_ports,
    read_pci_device_list,
)
from simllm.placement.dgx import DGX_NVLINK_BUNDLES

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "nccl_topology" / "vastai_hgx_h200_8x"
B200_FIXTURE = ROOT / "tests" / "fixtures" / "nccl_topology" / "vastai_hgx_b200_8x"
EXPECTATIONS = ROOT / "examples" / "hgx_h200_switch_capture_v1" / "expectations.json"
NODE = "hgx-h200"
BUS_IDS = tuple(f"0000:{bus}:00.0" for bus in ("83", "8b", "93", "9b", "a3", "ab", "b3", "bb"))
PCIE_SWITCHES = tuple(f"0000:{bus}:00.0" for bus in ("81", "89", "91", "99", "a1", "a9", "b1", "b9"))
NUMA = (0, 0, 0, 0, 1, 1, 1, 1)
MODULE_IDS = (2, 4, 1, 3, 7, 5, 6, 8)
SLOTS = (1, 3, 0, 2, 6, 4, 5, 7)
NVSWITCHES = tuple(f"0000:{bus}:00.0" for bus in ("c3", "c4", "c5", "c6"))
LANES_PER_SWITCH = (4, 5, 5, 4)
PORTS_PER_SWITCH = (32, 40, 40, 32)
BOARD_IDS = ("0x8300", "0x8b00", "0x9300", "0x9b00", "0xa300", "0xab00", "0xb300", "0xbb00")
B200_PRESET_DIGEST = (125_741, "ef920aa9abc7b61222880d97b60d2e016b37ed1df46b5c628488054553f5163f")
DGX_RESULTS_SHA256 = "00d935fc4b26cfc50aa4c5f6b46ba20e3db856d091215c2e616b095fb577fdba"
B200_RESULTS_SHA256 = "778a66d642e071e83d642b16622e810cda30fed937c6cea70b061304418bc122"
REMOTE_BLOCK = "nvidia-smi nvlink -R (remote pci bus id per link)"
NVSWITCH_BLOCK = "pci devices class 0x0680 (nvswitch/bridge)"
MODULE_BLOCK = "nvidia-smi -q (GPU board/module ids)"


def fixture_bytes(name: str, directory: Path = FIXTURE) -> bytes:
    # A Windows checkout may rewrite line endings; the frozen digests are LF.
    return (directory / name).read_bytes().replace(b"\r\n", b"\n")


def load() -> NcclTopologyDump:
    return NcclTopologyDump.load(FIXTURE / "nccl_topo.xml")


def sections(directory: Path = FIXTURE) -> dict[str, tuple[str, ...]]:
    return read_inventory_sections(fixture_bytes("node_inventory.txt", directory).decode("utf-8"))


def switch_table(inventory=None) -> dict[int, tuple[tuple[str, int], ...]]:
    inventory = sections() if inventory is None else inventory
    return captured_switch_ports(read_nvlink_remote_ports(inventory[REMOTE_BLOCK]),
                                 read_pci_device_list(inventory[NVSWITCH_BLOCK]))


def modules() -> dict[int, int]:
    return read_module_ids(sections()[MODULE_BLOCK])


def join(dump: NcclTopologyDump | None = None, **overrides):
    arguments = {
        "generation": "h100",
        "node_id": NODE,
        "pool_role": "serving",
        "global_rank_by_gpu_dev": {dev: dev for dev in range(8)},
        "propagation_delay_ps": 1000,
        "switch_input_buffer_bytes": 65536,
        "module_id_by_gpu_dev": modules(),
        "switch_ports_by_gpu_dev": switch_table(),
    }
    arguments.update(overrides)
    return captured_switched_node(load() if dump is None else dump, **arguments)


def canonical_json(fabric) -> bytes:
    return json.dumps(asdict(fabric), sort_keys=True, separators=(",", ":")).encode()


def captured_pair(port_id: str) -> tuple[str, int]:
    match = re.search(r":switch-([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]):port-([0-9]+)$",
                      port_id)
    assert match is not None, port_id
    return match.group(1), int(match.group(2))


def test_h1_parse_reproduces_the_captured_board():
    dump = load()
    switches = [device for device in dump.pci if device.pci_class == "0x060400"]
    assert (len(dump.cpus), len(switches), len(dump.gpus), len(dump.nvlinks)) == (2, 8, 8, 32)
    assert [cpu.numaid for cpu in dump.cpus] == [0, 1]
    assert {(device.busid, device.vendor, device.device) for device in switches} == {
        (busid, "0x104c", "0x8232") for busid in PCIE_SWITCHES
    }
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    devices = {device.busid: device for device in dump.pci}
    assert tuple(gpu.busid for gpu in gpus) == BUS_IDS
    assert tuple(devices[busid].parent_busid for busid in BUS_IDS) == PCIE_SWITCHES
    assert tuple(dump.numa_of(busid) for busid in BUS_IDS) == NUMA
    assert {(devices[busid].vendor, devices[busid].device, devices[busid].pci_class)
            for busid in BUS_IDS} == {("0x10de", "0x2335", "0x030200")}
    assert {(gpu.sm, gpu.gdr) for gpu in gpus} == {(90, 1)}
    assert {(row.tclass, row.switch_attached) for row in dump.nvlinks} == {("0x068000", True)}
    for busid in BUS_IDS:
        rows = dump.switch_rows(busid)
        assert sorted((row.target, row.count) for row in rows) == list(
            zip(NVSWITCHES, LANES_PER_SWITCH, strict=True))
        assert dump.switch_lanes(busid) == 18
    (socket,) = dump.distinct_nets
    assert (socket.name, socket.busid, socket.speed, socket.gdr) == ("eth0", None, 10_000, 0)
    assert [net.numaid for net in dump.nets] == [0, 1]
    assert fixture_bytes("nccl_env.txt").decode("utf-8").splitlines()[0] == (
        "torch 2.8.0+cu128 cuda 12.8 nccl (2, 27, 3)"
    )
    # NCCL graph patterns: 4 is ring, 1 is balanced tree, 5 is NVLS.
    graphs = {graph.get("pattern"): graph
              for graph in ElementTree.fromstring(fixture_bytes("nccl_graph.xml")).iter("graph")}
    assert [(graphs[pattern].get("nchannels"), graphs[pattern].get("typeintra"))
            for pattern in ("4", "1", "5")] == [("12", "NVL"), ("12", "NVL"), ("8", "NVL")]


def test_h1_switch_rows_keep_dump_order():
    dump = load()
    assert [row.target for row in dump.switch_rows(BUS_IDS[0])] == [
        "0000:c5:00.0", "0000:c4:00.0", "0000:c3:00.0", "0000:c6:00.0",
    ]
    assert [row.count for row in dump.switch_rows(BUS_IDS[0])] == [5, 5, 4, 4]


def test_h2_inventory_parsers_read_links_switches_and_module_ids():
    inventory = sections()
    remote = read_nvlink_remote_ports(inventory[REMOTE_BLOCK])
    assert sorted(remote) == list(range(8))
    assert sum(len(rows) for rows in remote.values()) == 144
    assert {len(rows) for rows in remote.values()} == {18}
    for rows in remote.values():
        counts = Counter(busid for busid, _ in rows)
        assert [counts[switch] for switch in NVSWITCHES] == list(LANES_PER_SWITCH)
    pairs = [pair for rows in remote.values() for pair in rows]
    assert len(set(pairs)) == 144
    per_switch = Counter(busid for busid, _ in set(pairs))
    assert [per_switch[switch] for switch in NVSWITCHES] == list(PORTS_PER_SWITCH)
    devices = read_pci_device_list(inventory[NVSWITCH_BLOCK])
    assert [(d.busid, d.pci_class, d.vendor, d.device, d.numa) for d in devices] == [
        (busid, "0x068000", "0x10de", "0x22a3", 1) for busid in NVSWITCHES
    ]
    found = read_module_ids(inventory[MODULE_BLOCK])
    assert tuple(found[dev] for dev in range(8)) == MODULE_IDS
    assert tuple(found[dev] - 1 for dev in range(8)) == SLOTS


def test_inventory_readers_refuse_what_they_would_have_to_guess():
    b200 = sections(B200_FIXTURE)
    with pytest.raises(ValueError, match="virtual fabric address"):
        read_nvlink_remote_ports(b200[REMOTE_BLOCK])
    lines = list(sections()[REMOTE_BLOCK])
    swapped = copy.copy(lines)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(ValueError, match="GPU 0 link 1 is out of order"):
        read_nvlink_remote_ports(swapped)
    with pytest.raises(ValueError, match="unreadable nvlink -R line"):
        read_nvlink_remote_ports([lines[0], "\t Link 0: Remote Device nowhere"])
    with pytest.raises(ValueError, match="repeats a bus id"):
        read_pci_device_list([*sections()[NVSWITCH_BLOCK], sections()[NVSWITCH_BLOCK][0]])
    with pytest.raises(ValueError, match="Minor Number and Module Id"):
        read_module_ids([line for line in sections()[MODULE_BLOCK] if "Module Id" not in line])


def test_h3_bundle_equals_captured_counts_and_ports_are_bijective():
    table = switch_table()
    bundle = DGX_NVLINK_BUNDLES["h100"]
    for rows in table.values():
        counts = Counter(busid for busid, _ in rows)
        assert tuple(counts[switch] for switch in sorted(counts)) == bundle == LANES_PER_SWITCH
    _, bound = join()
    switch_ports = [port for port in bound.ports if port.switch_id is not None]
    assert len(switch_ports) == 144
    bound_pairs = [captured_pair(port.port_id) for port in switch_ports]
    assert len(set(bound_pairs)) == 144
    assert set(bound_pairs) == {pair for rows in table.values() for pair in rows}
    assert sorted({port.switch_id for port in switch_ports}) == [
        f"{NODE}:hgx-h100-8:switch-{busid}" for busid in NVSWITCHES
    ]
    assert all(port.port_id.startswith(port.switch_id + ":port-") for port in switch_ports)


def test_h3_every_path_joins_two_ports_of_its_preset_chip_switch():
    _, bound = join()
    _, declared = join(switch_ports_by_gpu_dev=None)
    for source, destination in permutations(range(8), 2):
        bound_paths = bound.paths_between(source, destination)
        declared_paths = declared.paths_between(source, destination)
        assert len(bound_paths) == len(declared_paths) == 18
        for bound_path, declared_path in zip(bound_paths, declared_paths, strict=True):
            assert bound_path.switch_input_port_id.startswith(bound_path.switch_id + ":")
            assert bound_path.switch_output_port_id.startswith(bound_path.switch_id + ":")
            chip = int(declared_path.switch_id.rsplit("switch-", 1)[1])
            assert bound_path.switch_id.endswith(NVSWITCHES[chip - 1])
            assert bound_path.input_link.link_id == declared_path.input_link.link_id
            assert bound_path.output_link.link_id == declared_path.output_link.link_id


def test_h3_lane_k_binds_the_kth_captured_link_and_totals_conserve():
    table = switch_table()
    _, bound = join()
    owner = {port.port_id: port.gpu_rank for port in bound.ports}
    per_switch = Counter()
    for link in bound.links:
        slot, chip, lane = map(int, re.search(r":link-([0-9]+)-([0-9]+)-([0-9]+)$",
                                              link.link_id).groups())
        dev = SLOTS.index(slot)
        assert owner[link.endpoint_a] == dev
        busid, port = captured_pair(link.endpoint_b)
        assert busid == NVSWITCHES[chip - 1]
        assert port == [p for b, p in table[dev] if b == busid][lane]
        per_switch[busid] += 1
    assert [per_switch[switch] for switch in NVSWITCHES] == list(PORTS_PER_SWITCH)


def test_h3_without_the_table_the_join_is_the_b200_slice_join():
    _, unbound = join(switch_ports_by_gpu_dev=None)
    preset = dgx_peer_fabric("h100", node_id=NODE, ranks=tuple(SLOTS.index(slot) for slot in range(8)),
                             propagation_delay_ps=1000, switch_input_buffer_bytes=65536)
    assert canonical_json(unbound) == canonical_json(preset)
    _, bound = join()
    assert [port.port_id for port in bound.ports if port.gpu_rank is not None] == [
        port.port_id for port in unbound.ports if port.gpu_rank is not None]
    assert [link.link_id for link in bound.links] == [link.link_id for link in unbound.links]
    assert bound.routes == unbound.routes
    assert [(link.link_rate_bps, link.propagation_delay_ps) for link in bound.links] == [
        (link.link_rate_bps, link.propagation_delay_ps) for link in unbound.links]


def _query_records(title: str) -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    for line in sections()[title]:
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9 ]*?)\s*:\s*(.*?)\s*", line)
        if match is None:
            continue
        key, value = match.groups()
        if key == "Product Name":
            records.append({})
        if records:
            records[-1].setdefault(key, []).append(value)
    return records


def test_h4_gpu_side_matches_nvidia_smi():
    dump = load()
    node, bound = join()
    status: dict[int, list[str]] = {}
    for line in sections()["nvidia-smi nvlink -s"]:
        heading = re.match(r"GPU ([0-9]+): ", line)
        if heading:
            status[int(heading.group(1))] = []
        elif line.strip():
            status[max(status)].append(re.fullmatch(r"\s*Link [0-9]+: (.+)", line).group(1))
    assert status == {dev: ["26.562 GB/s"] * 18 for dev in range(8)}
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    matrix = [ansi.sub("", line) for line in sections()["nvidia-smi topo -m"]]
    numa_column = matrix[0].split("\t").index("NUMA Affinity")
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in matrix if re.match(r"GPU[0-9]+\t", line)}
    for source, destination in permutations(range(8), 2):
        assert rows[source][1 + destination].strip() == "NV18"
        assert len(bound.paths_between(source, destination)) == 18
    assert [dump.switch_lanes(busid) for busid in BUS_IDS] == [18] * 8
    listed = [re.fullmatch(r"GPU ([0-9]+): NVIDIA H200 \(UUID: (GPU-[0-9a-f-]+)\)", line)
              for line in sections()["nvidia-smi -L"]]
    full = _query_records("nvidia-smi -q full")
    assert [int(match.group(1)) for match in listed] == list(range(8))
    assert [record["GPU UUID"][0] for record in full] == [match.group(2) for match in listed]
    assert [record["Bus Id"][0].lower().replace("00000000:", "0000:") for record in full] == list(
        BUS_IDS)
    assert tuple(gpu.gpu_id for gpu in node.gpus) == BUS_IDS
    assert len({device.parent_busid for device in dump.pci if device.busid in BUS_IDS}) == 8
    assert [int(rows[dev][numa_column]) for dev in range(8)] == list(NUMA)
    assert Counter(NUMA) == {0: 4, 1: 4}
    board = _query_records(MODULE_BLOCK)
    assert tuple(record["Board ID"][0] for record in board) == BOARD_IDS
    assert {part for record in full for part in record.get("Board Part Number", [])} == {
        "695-2G520-0280-001"}
    for port in bound.ports:
        if port.gpu_rank is not None:
            assert f":gpu-{SLOTS[port.gpu_rank]}:" in port.port_id


def _five_lanes_on_c3(table):
    rows = list(table[0])
    index = next(i for i, (busid, _) in enumerate(rows) if busid == "0000:c6:00.0")
    rows[index] = ("0000:c3:00.0", 99)
    return {**table, 0: tuple(rows)}


def _port_used_twice(table):
    shared = next(pair for pair in table[0] if pair[0] == "0000:c3:00.0")
    rows = list(table[1])
    index = next(i for i, (busid, _) in enumerate(rows) if busid == "0000:c3:00.0")
    rows[index] = shared
    return {**table, 1: tuple(rows)}


def _virtual_row_dump():
    root = ElementTree.fromstring(fixture_bytes("nccl_topo.xml"))
    root.find(".//gpu[@dev='0']").append(ElementTree.Element(
        "nvlink", {"target": "fffffff:ff:ff.0", "count": "18", "tclass": "0x068000"}))
    return ElementTree.tostring(root, encoding="unicode")


def _three_switches():
    inventory = dict(sections())
    inventory[NVSWITCH_BLOCK] = tuple(line for line in inventory[NVSWITCH_BLOCK]
                                      if not line.startswith("0000:c6:00.0"))
    return switch_table(inventory)


def _seventeen_links():
    inventory = dict(sections())
    lines = list(inventory[REMOTE_BLOCK])
    heading = next(i for i, line in enumerate(lines) if line.startswith("GPU 3:"))
    del lines[heading + 18]
    inventory[REMOTE_BLOCK] = tuple(lines)
    return switch_table(inventory)


REFUSALS = {
    "5 lanes on c3": (
        lambda: join(switch_ports_by_gpu_dev=_five_lanes_on_c3(switch_table())),
        (r"GPU dev 0 has 5 captured lanes on switch 0000:c3:00\.0; chip 1 of generation h100 "
         r"requires 4"),
    ),
    "switch port used twice": (
        lambda: join(switch_ports_by_gpu_dev=_port_used_twice(switch_table())),
        r"captured switch 0000:c3:00\.0 port [0-9]+ is used by two links",
    ),
    "virtual row beside real rows": (
        lambda: join(NcclTopologyDump.parse(_virtual_row_dump())),
        r"gpu 0000:83:00\.0 mixes the virtual NVSwitch row with real switch rows",
    ),
    "three switches": (
        lambda: join(switch_ports_by_gpu_dev=_three_switches()),
        r"GPU 0 link 12 reaches 0000:c6:00\.0, which is not a listed NVSwitch",
    ),
    "b200 against four switches": (
        lambda: join(generation="b200"),
        r"generation b200 binds 2 NVSwitch chips; the captured links name 4 switches",
    ),
    "17 links for one gpu": (
        lambda: join(switch_ports_by_gpu_dev=_seventeen_links()),
        r"GPU dev 3 has 17 captured NVLink links; generation h100 requires 18",
    ),
}


@pytest.fixture
def no_schema_objects(monkeypatch):
    def refuse_build(*_args, **_kwargs):
        raise AssertionError("a schema object was built before the refusal")

    for name in ("FabricNodePlacement", "GpuFabricPlacement", "dgx_peer_fabric"):
        monkeypatch.setattr(nccl_topology, name, refuse_build)


def test_h6_refusal_family_matches_the_freeze():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["cells"]["h6_refusals"]
    assert sorted(REFUSALS) == sorted(frozen)


@pytest.mark.parametrize("label", sorted(REFUSALS))
def test_h6_refusal_precedes_every_schema_object(label, no_schema_objects):
    build, message = REFUSALS[label]
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"switch_ports_by_gpu_dev": {dev: [("0000:c3:00.0", "1")] * 18 for dev in range(8)}},
         TypeError, "pairs"),
        ({"switch_ports_by_gpu_dev": {dev: () for dev in range(7)}}, ValueError,
         "must name exactly the GPU devs"),
    ],
)
def test_switch_side_refuses_malformed_tables(overrides, error, message, no_schema_objects):
    with pytest.raises(error, match=message):
        join(**overrides)


def test_preset_refuses_half_a_captured_switch_side():
    with pytest.raises(ValueError, match="given together"):
        dgx_peer_fabric("h100", node_id=NODE, ranks=tuple(range(8)), propagation_delay_ps=0,
                        switch_input_buffer_bytes=272, switch_ids=("a", "b", "c", "d"))


def test_h7_switch_bound_fabric_manifest_round_trips(tmp_path):
    node, bound = join()
    fabric = FabricTopologyManifest(nodes=[node], peer_fabrics=(bound,), source="extracted")
    fabric.validate()
    first = fabric.save(tmp_path / "fabric.json")
    loaded = FabricTopologyManifest.load(first)
    assert loaded == fabric
    assert loaded.save(tmp_path / "again.json").read_bytes() == first.read_bytes()


@pytest.mark.parametrize("generation", ("a100", "h100", "b200"))
def test_h7_preset_digests_are_unchanged(generation):
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["baseline"]["artifacts"]
    fabric = dgx_peer_fabric(generation, node_id="node-0", ranks=tuple(range(8)),
                             propagation_delay_ps=1000, switch_input_buffer_bytes=65536)
    digest = hashlib.sha256(canonical_json(fabric)).hexdigest()
    if generation == "b200":
        assert (len(canonical_json(fabric)), digest) == B200_PRESET_DIGEST
    else:
        assert digest == frozen[f"dgx_{generation}_peer_fabric_json"]


def test_h7_both_switched_study_results_are_unchanged():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["baseline"]["artifacts"]
    assert frozen["dgx_nvlink_v1_results"] == DGX_RESULTS_SHA256
    assert hashlib.sha256(
        (ROOT / "examples" / "dgx_nvlink_v1" / "results.json").read_bytes()
        .replace(b"\r\n", b"\n")).hexdigest() == DGX_RESULTS_SHA256
    assert hashlib.sha256(
        (ROOT / "examples" / "hgx_b200_capture_v1" / "results.json").read_bytes()
        .replace(b"\r\n", b"\n")).hexdigest() == B200_RESULTS_SHA256


def test_freeze_pins_the_h200_expectations():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    assert (frozen["schema"], frozen["tasks"], frozen["date"]) == (
        "simllm-hgx-h200-switch-capture-expectations-v1", ["PLACE-12", "PLACE-6"], "2026-09-13",
    )
    assert frozen["baseline"]["commit"] == "af5c0110ae74f69afa69cfec0b42281e123b3170"
    assert frozen["interface"]["join_keyword"] == "switch_ports_by_gpu_dev"
    assert frozen["interface"]["switch_port_identity"] == "<domain>:switch-<busid>:port-<n>"
    assert frozen["interface"]["switch_identity"] == "<domain>:switch-<busid>"
    capture = frozen["capture"]
    assert capture["nvswitch"]["busids"] == list(NVSWITCHES)
    assert capture["nvlink"]["lanes_per_switch_per_gpu"] == {"c3": 4, "c4": 5, "c5": 5, "c6": 4}
    assert capture["nvlink"]["ports_per_switch"] == {"c3": 32, "c4": 40, "c5": 40, "c6": 32}
    assert capture["gpu"]["module_id_by_dev"] == {str(dev): m for dev, m in enumerate(MODULE_IDS)}
    assert capture["gpu"]["slot_by_dev"] == {str(dev): s for dev, s in enumerate(SLOTS)}
    assert capture["gpu"]["board_ids"] == list(BOARD_IDS)
    assert frozen["cells"]["h2_inventory"]["slots"] == list(SLOTS)
    assert sorted(frozen["cells"]) == ["h1_parse", "h2_inventory", "h3_switch_side",
                                       "h4_gpu_side", "h5_live_identity", "h6_refusals",
                                       "h7_off_path"]
    for name, digest in frozen["baseline"]["fixture"].items():
        if name != "directory":
            assert hashlib.sha256(fixture_bytes(name)).hexdigest() == digest, name
