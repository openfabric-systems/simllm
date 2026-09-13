"""HGX B200 preset and the switched NCCL topology shape (PLACE-6, first slice).

Cells G1, G2, G3, G5 and G6 of examples/hgx_b200_capture_v1/expectations.md,
as amended by expectations-amendment-2026-09-13.md and
expectations-amendment-2-2026-09-13.md, plus freeze pins on the JSON
registries. Cell G4 and the DGX study's check run in the study harness.

The b200 preset digest pinned below (125,741 bytes, SHA-256
ef920aa9abc7b61222880d97b60d2e016b37ed1df46b5c628488054553f5163f) was computed
by this implementation after the freeze. It is a post-implementation
regression pin, not a frozen value.
"""

from __future__ import annotations

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
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    NcclTopologyDump,
    captured_fabric_node,
    captured_switched_node,
    dgx_peer_fabric,
    nccl_topology,
)
from simllm.placement.dgx import DGX_GPU_SILICON, DGX_NVLINK_BUNDLES

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "nccl_topology" / "vastai_hgx_b200_8x"
STUDY = ROOT / "examples" / "hgx_b200_capture_v1"
EXPECTATIONS = STUDY / "expectations.json"
AMENDMENT = STUDY / "expectations-amendment-2026-09-13.json"
AMENDMENT_2 = STUDY / "expectations-amendment-2-2026-09-13.json"
NODE = "hgx-b200"
BUS_IDS = tuple(f"0000:{bus}:00.0" for bus in ("51", "52", "62", "63", "75", "76", "86", "87"))
SWITCHES = tuple(f"0000:{bus}:00.0" for bus in ("45", "45", "56", "56", "67", "67", "7a", "7a"))
NUMA = (0, 0, 0, 0, 1, 1, 1, 1)
MODULE_IDS = (4, 2, 1, 3, 8, 6, 5, 7)
SLOTS = (3, 1, 0, 2, 7, 5, 4, 6)
B200_DIGEST = (125_741, "ef920aa9abc7b61222880d97b60d2e016b37ed1df46b5c628488054553f5163f")


def fixture_bytes(name: str) -> bytes:
    # A Windows checkout may rewrite line endings; the frozen digests are LF.
    return (FIXTURE / name).read_bytes().replace(b"\r\n", b"\n")


def load() -> NcclTopologyDump:
    return NcclTopologyDump.load(FIXTURE / "nccl_topo.xml")


def join(dump: NcclTopologyDump | None = None, **overrides):
    arguments = {
        "generation": "b200",
        "node_id": NODE,
        "pool_role": "serving",
        "global_rank_by_gpu_dev": {dev: dev for dev in range(8)},
        "propagation_delay_ps": 1000,
        "switch_input_buffer_bytes": 65536,
    }
    arguments.update(overrides)
    return captured_switched_node(load() if dump is None else dump, **arguments)


def preset(generation: str):
    return dgx_peer_fabric(generation, node_id="node-0", ranks=tuple(range(8)),
                           propagation_delay_ps=1000, switch_input_buffer_bytes=65536)


def canonical_json(fabric) -> bytes:
    return json.dumps(asdict(fabric), sort_keys=True, separators=(",", ":")).encode()


def inventory_section(title: str) -> list[str]:
    lines = fixture_bytes("node_inventory.txt").decode("utf-8").splitlines()
    start = lines.index(f"--- {title} ---") + 1
    end = next(
        (index for index in range(start, len(lines))
         if lines[index].startswith("--- ") and lines[index].endswith(" ---")),
        len(lines),
    )
    return lines[start:end]


def smi_bus_id(value: str) -> str:
    """nvidia-smi writes an eight-digit PCI domain in upper case."""

    domain, rest = value.split(":", 1)
    return f"{int(domain, 16):04x}:{rest.lower()}"


def smi_query(title: str) -> list[dict[str, list[str]]]:
    """One record per GPU from an nvidia-smi -q style block, split at Product Name."""

    records: list[dict[str, list[str]]] = []
    for line in inventory_section(title):
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9 ]*?)\s*:\s*(.*?)\s*", line)
        if match is None:
            continue
        key, value = match.groups()
        if key == "Product Name":
            records.append({})
        if records:
            records[-1].setdefault(key, []).append(value)
    return records


def module_ids() -> dict[int, int]:
    rows = smi_query("nvidia-smi -q (GPU board/module ids)")
    return {int(row["Minor Number"][0]): int(row["Module Id"][0]) for row in rows}


def topo_matrix() -> tuple[dict[int, list[str]], dict[int, int]]:
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi.sub("", line) for line in inventory_section("nvidia-smi topo -m")]
    numa_column = lines[0].split("\t").index("NUMA Affinity")
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in lines if re.match(r"GPU[0-9]+\t", line)}
    matrix = {dev: [cell.strip() for cell in row[1:9]] for dev, row in rows.items()}
    return matrix, {dev: int(row[numa_column]) for dev, row in rows.items()}


def nvlink_status() -> dict[int, list[str]]:
    links: dict[int, list[str]] = {}
    current = None
    for line in inventory_section("nvidia-smi nvlink -s"):
        heading = re.match(r"GPU ([0-9]+): ", line)
        if heading:
            current = int(heading.group(1))
            links[current] = []
        elif line.strip():
            links[current].append(re.fullmatch(r"\s*Link [0-9]+: (.+)", line).group(1))
    return links


def test_g1_parse_reproduces_the_captured_board():
    dump = load()
    switches = [device for device in dump.pci if device.pci_class == "0x060400"]
    assert (len(dump.cpus), len(switches), len(dump.gpus)) == (2, 4, 8)
    assert [cpu.numaid for cpu in dump.cpus] == [0, 1]
    assert {(device.busid, device.vendor, device.parent_busid) for device in switches} == {
        (busid, "0x1000", None) for busid in set(SWITCHES)
    }
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    assert tuple(gpu.busid for gpu in gpus) == BUS_IDS
    assert all(gpu.sm == 100 for gpu in gpus)
    devices = {device.busid: device for device in dump.pci}
    assert {(devices[busid].vendor, devices[busid].device, devices[busid].pci_class)
            for busid in BUS_IDS} == {("0x10de", "0x2901", "0x030200")}
    assert tuple(devices[busid].parent_busid for busid in BUS_IDS) == SWITCHES
    assert tuple(dump.numa_of(busid) for busid in BUS_IDS) == NUMA
    assert len(dump.nvlinks) == 8
    assert {(row.target, row.count, row.tclass, row.switch_attached) for row in dump.nvlinks} == {
        ("fffffff:ff:ff.0", 18, "0x068000", True)
    }
    assert sorted(row.source_busid for row in dump.nvlinks) == sorted(BUS_IDS)
    assert [dump.switch_lanes(busid) for busid in BUS_IDS] == [18] * 8
    (socket,) = dump.distinct_nets
    assert [net.numaid for net in dump.nets] == [0, 1]
    assert (socket.name, socket.busid, socket.speed, socket.gdr, socket.guid, socket.port) == (
        "eth0", None, 10_000, 0, "0x0", 0,
    )
    assert fixture_bytes("nccl_env.txt").decode("utf-8").splitlines()[0] == (
        "torch 2.8.0+cu128 cuda 12.8 nccl (2, 27, 3)"
    )
    # NCCL graph patterns: 4 is ring, 1 is balanced tree, 5 is NVLS.
    graphs = {graph.get("pattern"): graph
              for graph in ElementTree.fromstring(fixture_bytes("nccl_graph.xml")).iter("graph")}
    assert [(graphs[pattern].get("nchannels"), graphs[pattern].get("typeintra"))
            for pattern in ("4", "1", "5")] == [("16", "NVL"), ("16", "NVL"), ("8", "NVL")]


def test_switched_shape_leaves_the_pcie_path_and_socket_nic_readable():
    dump = load()
    assert dump.pcie_path("0000:87:00.0") == ("0000:7a:00.0", "0000:87:00.0")
    assert dump.pcie_location("0000:51:00.0") == "numa-0/pci-0000:45:00.0/pci-0000:51:00.0"
    assert dump.switch_lanes("0000:45:00.0") == 0


def test_g2_b200_preset_structure_and_post_implementation_digest():
    fabric = preset("b200")
    chips = {port.switch_id for port in fabric.ports if port.switch_id is not None}
    assert DGX_NVLINK_BUNDLES["b200"] == (9, 9)
    assert len(chips) == 2
    assert (len(fabric.links), len(fabric.ports), len(fabric.routes)) == (144, 288, 56)
    assert {link.link_rate_bps for link in fabric.links} == {400_000_000_000}
    for source, destination in permutations(range(8), 2):
        paths = fabric.paths_between(source, destination)
        assert len(paths) == 18
        assert Counter(path.switch_id for path in paths) == dict.fromkeys(chips, 9)
        assert len({path.input_resource for path in paths}) == 18
        assert len({path.output_resource for path in paths}) == 18
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}")
        for rank in range(8)
    ), ())
    FabricTopologyManifest(nodes=[node], peer_fabrics=(fabric,)).validate()
    payload = canonical_json(fabric)
    assert (len(payload), hashlib.sha256(payload).hexdigest()) == B200_DIGEST


def test_unknown_generation_names_every_supported_one():
    with pytest.raises(ValueError, match="a100, h100, b200"):
        preset("gb200")


def test_g3_lanes_per_gpu_equal_the_active_link_count():
    lanes = sum(DGX_NVLINK_BUNDLES["b200"])
    per_rank = Counter(port.gpu_rank for port in preset("b200").ports if port.gpu_rank is not None)
    assert set(per_rank.values()) == {lanes} == {18}
    status = nvlink_status()
    assert sorted(status) == list(range(8))
    assert all(rates == ["53.125 GB/s"] * lanes for rates in status.values())


def test_g3_paths_per_pair_equal_nv18_and_the_nccl_count():
    matrix, _ = topo_matrix()
    fabric = preset("b200")
    for source, destination in permutations(range(8), 2):
        cell = matrix[source][destination]
        assert cell == "NV18"
        assert len(fabric.paths_between(source, destination)) == int(cell[2:]) == 18
    assert all(matrix[dev][dev] == "X" for dev in range(8))
    assert [row.count for row in load().nvlinks] == [18] * 8


def test_g3_device_order_bus_ids_and_pcie_placement():
    node, _ = join()
    listed = {}
    for line in inventory_section("nvidia-smi -L"):
        match = re.fullmatch(r"GPU ([0-9]+): NVIDIA B200 \(UUID: (GPU-[0-9a-f-]+)\)", line)
        assert match is not None, line
        listed[int(match.group(1))] = match.group(2)
    rows = smi_query("nvidia-smi -q full")
    assert [int(row["Minor Number"][0]) for row in rows] == list(range(8)) == sorted(listed)
    assert [row["GPU UUID"][0] for row in rows] == [listed[dev] for dev in range(8)]
    assert tuple(smi_bus_id(row["Bus Id"][0]) for row in rows) == BUS_IDS
    assert tuple(gpu.gpu_id for gpu in node.gpus) == BUS_IDS
    assert [gpu.global_rank for gpu in node.gpus] == list(range(8))
    assert [gpu.pcie_location for gpu in node.gpus] == [
        f"numa-{numa}/pci-{switch}/pci-{busid}"
        for numa, switch, busid in zip(NUMA, SWITCHES, BUS_IDS, strict=True)
    ]
    assert all(SWITCHES[dev] == SWITCHES[dev + 1] for dev in range(0, 8, 2))
    assert len(set(SWITCHES)) == 4
    _, numa_column = topo_matrix()
    assert tuple(numa_column[dev] for dev in range(8)) == NUMA
    assert Counter(NUMA) == {0: 4, 1: 4}
    listed_numa = {}
    for line in inventory_section("nvidia pci devices with class and numa"):
        busid, device_class, numa = re.fullmatch(r"(\S+) class=(\S+) numa=([0-9]+)", line).groups()
        assert device_class == "0x030200"
        listed_numa[busid] = int(numa)
    assert tuple(listed_numa[busid] for busid in BUS_IDS) == NUMA
    assert node.nics == ()
    assert [gpu.nic_id for gpu in node.gpus] == [f"{NODE}:nic-absent-{dev}" for dev in range(8)]


def test_g3_board_identity_has_distinct_board_ids_and_one_part_number():
    rows = smi_query("nvidia-smi -q full")
    board_ids = [row["Board ID"][0] for row in rows]
    assert board_ids == ["0x5100", "0x5200", "0x6200", "0x6300", "0x7500", "0x7600", "0x8600",
                         "0x8700"]
    assert len(set(board_ids)) == 8
    assert {part for row in rows for part in row["Board Part Number"]} == {"692-2G525-0220-500"}


def test_g3_module_ids_bind_devices_to_preset_slots():
    modules = module_ids()
    assert tuple(modules[dev] for dev in range(8)) == MODULE_IDS
    assert [int(row["Module Id"][0]) for row in smi_query("nvidia-smi -q full")] == list(MODULE_IDS)
    _, fabric = join(module_id_by_gpu_dev=modules)
    slots: dict[int, set[int]] = {}
    for port in fabric.ports:
        if port.gpu_rank is not None:
            slots.setdefault(port.gpu_rank, set()).add(
                int(re.search(r":gpu-([0-9]+):", port.port_id).group(1)))
    assert slots == {dev: {SLOTS[dev]} for dev in range(8)}
    owner = {port.port_id: port.gpu_rank for port in fabric.ports}
    for link in fabric.links:
        slot = int(re.search(r":link-([0-9]+)-", link.link_id).group(1))
        (gpu_port,) = [end for end in (link.endpoint_a, link.endpoint_b) if owner[end] is not None]
        assert f":gpu-{slot}:" in gpu_port
        assert SLOTS[owner[gpu_port]] == slot
    _, unbound = join()
    assert SLOTS != tuple(range(8))
    assert [port.port_id for port in fabric.ports] == [port.port_id for port in unbound.ports]
    assert [port.gpu_rank for port in fabric.ports] != [port.gpu_rank for port in unbound.ports]


def test_g3_switch_side_is_recorded_as_not_observable():
    remote = [re.search(r"Remote Device ([0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+)",
                        line).group(1)
              for line in inventory_section("nvidia-smi nvlink -R (remote pci bus id per link)")
              if "Remote Device" in line]
    assert len(remote) == 8 * 18
    assert set(remote) == {"FFFFFFFF:FF:FF.0"}
    assert not any(line.strip()
                   for line in inventory_section("pci devices class 0x0680 (nvswitch/bridge)"))
    assert not any(line.strip() for line in inventory_section("nvswitch devices"))
    assert {"mlx5_0", "mlx5_1"} <= set(inventory_section("net devices"))


def mutate(change) -> str:
    root = ElementTree.fromstring(fixture_bytes("nccl_topo.xml"))
    change(root)
    return ElementTree.tostring(root, encoding="unicode")


def _seven_gpus(root):
    switch = root.find(".//pci[@busid='0000:7a:00.0']")
    switch.remove(switch.find("pci[@busid='0000:87:00.0']"))


def _mixed_rows(root):
    root.find(".//gpu[@dev='0']").append(ElementTree.Element(
        "nvlink", {"target": "0000:52:00.0", "count": "2", "tclass": "0x030200"}))


def _differing_socket_nic(root):
    root.findall("cpu/nic/net")[1].set("speed", "25000")


REFUSALS = {
    "17 lanes on one gpu": (
        lambda root: root.find(".//gpu[@dev='3']/nvlink").set("count", "17"), {},
        r"GPU dev 3 at 0000:63:00.0 has 17 switch-attached NVLink lanes; generation b200 requires 18",
    ),
    "seven gpus": (_seven_gpus, {"global_rank_by_gpu_dev": {dev: dev for dev in range(7)}},
                   r"requires eight GPUs; the dump holds 7"),
    "mixed switch and peer rows": (
        _mixed_rows, {}, r"gpu 0000:51:00.0 mixes switch-attached and peer-attached nvlink rows",
    ),
    "socket nic repeated with differing attributes": (
        _differing_socket_nic, {},
        r"socket net eth0 repeats under several cpus with differing attributes",
    ),
    "nested pci under non-switch": (
        lambda root: root.find(".//pci[@busid='0000:45:00.0']").set("class", "0x030200"), {},
        (r"pci 0000:45:00.0 of class 0x030200 holds a nested pci; only a PCIe switch "
         r"\(class 0x060400\) may"),
    ),
    "generation a100 against eighteen lanes": (
        None, {"generation": "a100"},
        r"has 18 switch-attached NVLink lanes; generation a100 requires 12",
    ),
    "module id map not a bijection onto 1..8": (
        None, {"module_id_by_gpu_dev": {**dict(enumerate(MODULE_IDS)), 7: 4}},
        r"module id map not a bijection onto 1\.\.8",
    ),
}


@pytest.fixture
def no_schema_objects(monkeypatch):
    def refuse_build(*_args, **_kwargs):
        raise AssertionError("a schema object was built before the refusal")

    for name in ("FabricNodePlacement", "GpuFabricPlacement", "dgx_peer_fabric"):
        monkeypatch.setattr(nccl_topology, name, refuse_build)


def test_g5_refusal_family_matches_the_freeze_and_amendment():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["cells"]["g5_refusals"]
    added = json.loads(AMENDMENT.read_text(encoding="utf-8"))["added"]["g5_refusal"]
    assert sorted(REFUSALS) == sorted([*frozen, added])


@pytest.mark.parametrize("label", sorted(REFUSALS))
def test_g5_refusal_precedes_every_schema_object(label, no_schema_objects):
    change, overrides, message = REFUSALS[label]
    with pytest.raises(ValueError, match=message):
        dump = load() if change is None else NcclTopologyDump.parse(mutate(change))
        join(dump, **overrides)


def test_g5_h100_generation_is_refused_on_silicon(no_schema_objects):
    with pytest.raises(ValueError, match=(
        r"GPU dev 0 at 0000:51:00\.0 reports device 0x10de:0x2901 with sm 100; "
        r"generation h100 requires device 0x2330 or 0x2335 or 0x2339 with sm 90"
    )):
        join(generation="h100")


@pytest.mark.parametrize(
    ("change", "overrides", "error", "message"),
    [
        (lambda root: [net.set("gdr", "1") for net in root.findall("cpu/nic/net")], {},
         ValueError, r"NIC eth0 supports GPU Direct RDMA"),
        (None, {"generation": "gb200"}, ValueError, r"generation must be one of a100, h100, b200"),
        (None, {"module_id_by_gpu_dev": [4, 2, 1, 3, 8, 6, 5, 7]}, TypeError, r"mapping"),
        (None, {"module_id_by_gpu_dev": dict(enumerate(range(1, 9))) | {8: 9}}, ValueError,
         r"module id map not a bijection"),
    ],
)
def test_switched_join_refuses_other_unmodeled_inputs(change, overrides, error, message,
                                                     no_schema_objects):
    dump = load() if change is None else NcclTopologyDump.parse(mutate(change))
    with pytest.raises(error, match=message):
        join(dump, **overrides)


def test_direct_mesh_join_refuses_the_switched_board():
    with pytest.raises(ValueError, match="GPU dev 0 at 0000:51:00.0 is switch-attached"):
        captured_fabric_node(load(), node_id=NODE, pool_role="serving",
                             global_rank_by_gpu_dev={dev: dev for dev in range(8)},
                             nvlink_propagation_delay_ps=1000)


def test_g6_joined_fabric_manifest_round_trips(tmp_path):
    node, mesh = join(module_id_by_gpu_dev=module_ids())
    fabric = FabricTopologyManifest(nodes=[node], peer_fabrics=(mesh,), source="extracted")
    fabric.validate()
    first = fabric.save(tmp_path / "fabric.json")
    loaded = FabricTopologyManifest.load(first)
    assert loaded == fabric
    assert loaded.save(tmp_path / "again.json").read_bytes() == first.read_bytes()


@pytest.mark.parametrize("generation", ("a100", "h100"))
def test_g6_accepted_preset_digest_is_unchanged(generation):
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["baseline"]["artifacts"]
    expected = frozen[f"dgx_{generation}_peer_fabric_json"]
    fabric = preset(generation)
    payload = canonical_json(fabric)
    assert len(payload) == expected["bytes"]
    assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    assert (len(fabric.links), len(fabric.ports), len(fabric.routes)) == (
        expected["links"], expected["ports"], expected["routes"],
    )
    assert {len(route.paths) for route in fabric.routes} == {expected["paths_per_route"]}
    assert {link.link_rate_bps for link in fabric.links} == {200_000_000_000}


def test_freeze_pins_the_expectations_and_the_amendment():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    assert (frozen["schema"], frozen["task"], frozen["date"]) == (
        "simllm-hgx-b200-capture-expectations-v1", "PLACE-6", "2026-09-13",
    )
    assert frozen["baseline"]["commit"] == "64c7f5923fc0fe0142d537a3bd4a054ee43ca02c"
    preset_block = dict(frozen["preset"])
    assert "Fabric Manager User Guide" in preset_block.pop("public_source")
    assert preset_block == {
        "generation": "b200", "nvlink_switch_chips": 2, "bundle": [9, 9], "lanes_per_gpu": 18,
        "link_payload_rate_bps": 400000000000, "signalling_gb_per_s_recorded": 53.125,
        "links": 144, "ports": 288, "routes": 56, "paths_per_route": 18,
        "paths_per_route_per_chip": 9,
    }
    assert frozen["interface"]["join"] == "captured_switched_node"
    assert frozen["interface"]["absent_nic_id"] == "<node_id>:nic-absent-<dev>"
    gpu = frozen["capture"]["gpu"]
    assert gpu["busid_by_dev"] == {str(dev): busid for dev, busid in enumerate(BUS_IDS)}
    assert gpu["pcie_switch_by_dev"] == {str(dev): switch for dev, switch in enumerate(SWITCHES)}
    assert gpu["numa_by_dev"] == {str(dev): numa for dev, numa in enumerate(NUMA)}
    assert frozen["capture"]["nvlink"]["nccl_row"] == {
        "target": "fffffff:ff:ff.0", "count": 18, "tclass": "0x068000",
    }
    assert sorted(frozen["cells"]) == ["g1_parse", "g2_preset", "g3_gpu_side", "g4_live",
                                       "g5_refusals", "g6_off_path"]
    assert json.loads(AMENDMENT.read_text(encoding="utf-8")) == {
        "schema": "simllm-hgx-b200-capture-amendment-v1",
        "date": "2026-09-13",
        "chronology": ("post-specified to a fixture reading error found in implementation "
                       "review, frozen before the harness and its first run"),
        "corrected": {
            "module_id_visible": True,
            "module_id_by_dev": {str(dev): module for dev, module in enumerate(MODULE_IDS)},
            "slot_by_dev": {str(dev): slot for dev, slot in enumerate(SLOTS)},
            "rdma_nics_nccl_visible": 0,
            "rdma_devices_in_container": ["mlx5_0", "mlx5_1"],
        },
        "added": {
            "join_keyword": "module_id_by_gpu_dev",
            "slot_rule": "module_id - 1",
            "g5_refusal": "module id map not a bijection onto 1..8",
        },
    }
    for name, digest in frozen["baseline"]["fixture"].items():
        if name != "directory":
            assert hashlib.sha256(fixture_bytes(name)).hexdigest() == digest, name


def _silicon(device: str, sm: int, lanes: int | None = None):
    def change(root):
        for pci in root.iter("pci"):
            if pci.get("class") == "0x030200":
                pci.set("device", device)
        for gpu in root.iter("gpu"):
            gpu.set("sm", str(sm))
            if lanes is not None:
                gpu.find("nvlink").set("count", str(lanes))
    return change


@pytest.mark.parametrize(
    ("generation", "device", "sm", "lanes"),
    [
        ("a100", "0x20b0", 80, 12),
        ("a100", "0x20b2", 80, 12),
        ("h100", "0x2330", 90, 18),
        ("h100", "0x2335", 90, 18),
        ("h100", "0x2339", 90, 18),
        ("b200", "0x2901", 100, 18),
    ],
)
def test_switched_join_accepts_each_generations_silicon(generation, device, sm, lanes):
    _, mesh = join(NcclTopologyDump.parse(mutate(_silicon(device, sm, lanes))),
                   generation=generation)
    assert mesh.domain_id == f"{NODE}:hgx-{generation}-8"


def test_switched_join_refuses_h200_silicon_against_b200(no_schema_objects):
    with pytest.raises(ValueError, match=(
        r"GPU dev 0 at 0000:51:00\.0 reports device 0x10de:0x2335 with sm 90; "
        r"generation b200 requires device 0x2901 with sm 100"
    )):
        join(NcclTopologyDump.parse(mutate(_silicon("0x2335", 90))))


def _gpu_outside_switch(root):
    cpu = root.find("cpu[@numaid='0']")
    switch = cpu.find("pci[@busid='0000:45:00.0']")
    gpu = switch.find("pci[@busid='0000:51:00.0']")
    switch.remove(gpu)
    cpu.insert(0, gpu)


def _nested_switch(root):
    cpu = root.find("cpu[@numaid='0']")
    switch = cpu.find("pci[@busid='0000:45:00.0']")
    outer = ElementTree.Element("pci", {
        "busid": "0000:44:00.0", "class": "0x060400", "vendor": "0x1000", "device": "0x0001",
        "link_speed": "32.0 GT/s PCIe", "link_width": "16",
    })
    cpu.remove(switch)
    outer.append(switch)
    cpu.insert(0, outer)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_gpu_outside_switch, r"GPU dev 0 at 0000:51:00\.0 is not under a PCIe switch"),
        (_nested_switch,
         r"PCIe switch 0000:45:00\.0 above GPU dev 0 is nested under PCIe switch 0000:44:00\.0"),
    ],
)
def test_switched_join_refuses_irregular_pcie_nesting(change, message, no_schema_objects):
    with pytest.raises(ValueError, match=message):
        join(NcclTopologyDump.parse(mutate(change)))


def test_second_amendment_pins_the_silicon_table():
    amendment = json.loads(AMENDMENT_2.read_text(encoding="utf-8"))
    assert amendment == {
        "schema": "simllm-hgx-b200-capture-amendment-2-v1",
        "date": "2026-09-13",
        "chronology": ("post-specified to an independent review finding, frozen before the "
                       "corrected join and its rerun"),
        "withdrawn": ["g5 h100 accepts eighteen lanes"],
        "silicon": {
            "a100": {"devices": ["0x20b0", "0x20b2"], "sm": 80},
            "h100": {"devices": ["0x2330", "0x2335", "0x2339"], "sm": 90},
            "b200": {"devices": ["0x2901"], "sm": 100},
        },
        "added_refusals": ["gpu outside a pcie switch", "pcie switch nested in a pcie switch",
                           "h100 against b200 silicon"],
    }
    assert {generation: {"devices": list(devices), "sm": sm}
            for generation, (devices, sm) in DGX_GPU_SILICON.items()} == amendment["silicon"]
