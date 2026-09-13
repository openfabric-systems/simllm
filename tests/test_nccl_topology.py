"""NCCL topology dump reader and captured-node join (PLACE-1 second slice).

Cells N1 to N4, N6 and N7 of examples/nccl_topology_capture_v1/expectations.md,
the eight fatal compatibility digests of its JSON registry, and a freeze pin on
that registry's schema and capture block. Cell N5 is live and runs in the
study harness.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
from xml.etree import ElementTree

import pytest

from simllm.placement import (
    FabricLink,
    FabricTopologyManifest,
    NcclTopologyDump,
    captured_fabric_node,
    declared_manifest,
    declared_pipeline_placement,
    dgx_peer_fabric,
    disaggregated_manifests,
    nccl_topology,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "nccl_topology" / "perlmutter_a100_nid001056"
EXPECTATIONS = ROOT / "examples" / "nccl_topology_capture_v1" / "expectations.json"
NODE = "nid001056"
GPU_BUS_IDS = ("0000:03:00.0", "0000:41:00.0", "0000:82:00.0", "0000:c1:00.0")
GPU_NUMA = (3, 2, 1, 0)
NICS_IN_NAME_ORDER = (
    ("cxi0", "0000:c2:00.0", 0, 3),
    ("cxi1", "0000:81:00.0", 1, 2),
    ("cxi2", "0000:42:00.0", 2, 1),
    ("cxi3", "0000:01:00.0", 3, 0),
)
#: PCI ID database names of the two captured device kinds, as lspci prints them.
LSPCI_IDS = {
    "NVIDIA Corporation GA100 [A100 SXM4 40GB]": ("0x10de", "0x20b0"),
    "Cray Inc Cassini 1 [Slingshot 200Gb]": ("0x17db", "0x0501"),
}
SCHEMA_BUILDERS = (
    "FabricLink",
    "FabricNodePlacement",
    "GpuFabricPlacement",
    "NicFabricPlacement",
    "PeerFabric",
    "PeerPortPlacement",
    "PeerRoute",
)


def fixture_bytes(name: str) -> bytes:
    # A Windows checkout may rewrite line endings; the frozen digests are LF.
    return (FIXTURE / name).read_bytes().replace(b"\r\n", b"\n")


def dump_text() -> str:
    return fixture_bytes("nccl_topo.xml").decode("utf-8")


def load() -> NcclTopologyDump:
    return NcclTopologyDump.load(FIXTURE / "nccl_topo.xml")


def join(dump: NcclTopologyDump, **overrides):
    arguments = {
        "node_id": NODE,
        "pool_role": "serving",
        "global_rank_by_gpu_dev": {dev: dev for dev in range(4)},
        "nvlink_propagation_delay_ps": 1000,
    }
    arguments.update(overrides)
    return captured_fabric_node(dump, **arguments)


def inventory_section(title: str) -> list[str]:
    lines = fixture_bytes("node_inventory.txt").decode("utf-8").splitlines()
    start = lines.index(f"--- {title} ---") + 1
    end = next(
        (index for index in range(start, len(lines))
         if lines[index].startswith("--- ") and lines[index].endswith(" ---")),
        len(lines),
    )
    return lines[start:end]


def test_n1_parse_reproduces_every_captured_literal():
    dump = load()
    assert dump.version == "1"
    assert (len(dump.cpus), len(dump.pci), len(dump.gpus), len(dump.nets), len(dump.nvlinks)) == (
        4, 8, 4, 4, 12,
    )
    assert [cpu.numaid for cpu in dump.cpus] == [3, 0, 1, 2]
    assert {(cpu.host_hash, cpu.arch, cpu.vendor) for cpu in dump.cpus} == {
        ("0x94ecb6d613fd2af1", "x86_64", "AuthenticAMD")
    }
    assert {(device.link_speed, device.link_width) for device in dump.pci} == {
        ("16.0 GT/s PCIe", 16)
    }
    gpu_busids = {gpu.busid for gpu in dump.gpus}
    assert {(d.vendor, d.device, d.pci_class) for d in dump.pci if d.busid in gpu_busids} == {
        ("0x10de", "0x20b0", "0x030200")
    }
    assert {(d.vendor, d.device, d.pci_class) for d in dump.pci if d.busid not in gpu_busids} == {
        ("0x17db", "0x0501", "0x020000")
    }
    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    assert [gpu.dev for gpu in gpus] == [0, 1, 2, 3]
    assert tuple(gpu.busid for gpu in gpus) == GPU_BUS_IDS
    assert tuple(dump.numa_of(gpu.busid) for gpu in gpus) == GPU_NUMA
    assert all((gpu.sm, gpu.rank, gpu.gdr) == (80, gpu.dev, 1) for gpu in gpus)
    for gpu in gpus:
        rows = [row for row in dump.nvlinks if row.source_busid == gpu.busid]
        assert len(rows) == 3
        assert {row.target for row in rows} == set(GPU_BUS_IDS) - {gpu.busid}
        assert {(row.count, row.tclass) for row in rows} == {(4, "0x030200")}
    assert [net.name for net in dump.nets] == ["cxi3", "cxi0", "cxi1", "cxi2"]
    assert [net.busid for net in dump.nets] == [
        "0000:01:00.0", "0000:c2:00.0", "0000:81:00.0", "0000:42:00.0",
    ]
    assert [dump.numa_of(net.busid) for net in dump.nets] == [3, 0, 1, 2]
    assert {(net.speed, net.speed_bps, net.port, net.gdr, net.maxconn) for net in dump.nets} == {
        (200_000, 200_000_000_000, 1, 1, 128)
    }
    affinity = {
        gpu.dev: next(net.name for net in dump.nets
                      if dump.numa_of(net.busid) == dump.numa_of(gpu.busid))
        for gpu in gpus
    }
    assert affinity == {0: "cxi3", 1: "cxi2", 2: "cxi1", 3: "cxi0"}


def test_n2_join_places_gpus_on_their_numa_affine_nic():
    node, _ = join(load())
    assert (node.node_id, node.pool_role) == (NODE, "serving")
    assert [(gpu.global_rank, gpu.gpu_id, gpu.node_id) for gpu in node.gpus] == [
        (rank, busid, NODE) for rank, busid in enumerate(GPU_BUS_IDS)
    ]
    assert [gpu.nic_id for gpu in node.gpus] == [
        "nid001056:cxi3", "nid001056:cxi2", "nid001056:cxi1", "nid001056:cxi0",
    ]
    assert [gpu.pcie_location for gpu in node.gpus] == [
        f"numa-{numa}/pci-{busid}" for numa, busid in zip(GPU_NUMA, GPU_BUS_IDS, strict=True)
    ]
    assert [(nic.nic_id, nic.node_id, nic.fabric_location, nic.affine_gpu_rank)
            for nic in node.nics] == [
        (f"{NODE}:{name}", NODE, f"numa-{numa}/pci-{busid}", rank)
        for name, busid, numa, rank in NICS_IN_NAME_ORDER
    ]
    assert all((nic.switch_id, nic.switch_port_id, nic.link_id) == (None, None, None)
               for nic in node.nics)


def test_join_takes_global_ranks_from_the_caller_not_the_dump():
    node, mesh = join(load(), global_rank_by_gpu_dev={0: 7, 1: 6, 2: 5, 3: 4})
    assert [gpu.global_rank for gpu in node.gpus] == [7, 6, 5, 4]
    assert [nic.affine_gpu_rank for nic in node.nics] == [4, 5, 6, 7]
    assert {(route.source_rank, route.destination_rank) for route in mesh.routes} == set(
        permutations((4, 5, 6, 7), 2)
    )
    # Physical names keep the device index; only rank ownership moves.
    assert mesh.ports[0].port_id == "nid001056:nv4:gpu-0:to-1:lane-0"
    assert mesh.ports[0].gpu_rank == 7


def test_n3_mesh_wires_every_pair_with_its_bonded_lanes():
    node, mesh = join(load(), nvlink_propagation_delay_ps=1234)
    domain = "nid001056:nv4"
    assert (mesh.domain_id, mesh.node_id, mesh.protocol, mesh.evidence_class) == (
        domain, NODE, "nvlink", "declared",
    )
    assert not mesh.switched
    assert mesh.switch_input_buffer_bytes == 65536
    assert (len(mesh.links), len(mesh.ports), len(mesh.routes)) == (24, 48, 12)
    assert sum(len(route.paths) for route in mesh.routes) == 48
    assert {(link.link_rate_bps, link.propagation_delay_ps) for link in mesh.links} == {
        (200_000_000_000, 1234)
    }
    assert mesh.links[0] == FabricLink(
        f"{domain}:link-0-1:lane-0", f"{domain}:gpu-0:to-1:lane-0",
        f"{domain}:gpu-1:to-0:lane-0", 200_000_000_000, 1234,
    )
    for source, destination in permutations(range(4), 2):
        rows = mesh.paths_between(source, destination)
        low, high = sorted((source, destination))
        assert len(rows) == 4
        assert len({row.input_resource for row in rows}) == 4
        assert all(row.output_link is None and row.switch_id is None for row in rows)
        assert [row.input_link.link_id for row in rows] == [
            f"{domain}:link-{low}-{high}:lane-{lane}" for lane in range(4)
        ]
        assert [(row.source_port_id, row.destination_port_id) for row in rows] == [
            (f"{domain}:gpu-{source}:to-{destination}:lane-{lane}",
             f"{domain}:gpu-{destination}:to-{source}:lane-{lane}")
            for lane in range(4)
        ]
    FabricTopologyManifest(nodes=[node], peer_fabrics=(mesh,), source="extracted").validate()


def test_join_passes_declared_rate_and_buffer_through():
    _, mesh = join(load(), nvlink_link_rate_bps=400_000_000_000, switch_input_buffer_bytes=4096)
    assert {link.link_rate_bps for link in mesh.links} == {400_000_000_000}
    assert mesh.switch_input_buffer_bytes == 4096


def test_n4_reader_agrees_with_the_node_inventory():
    dump = load()
    busid = {gpu.dev: gpu.busid for gpu in dump.gpus}

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    matrix = [ansi.sub("", line) for line in inventory_section("nvidia-smi topo -m")]
    header = matrix[0].split("\t")
    numa_column = header.index("NUMA Affinity")
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in matrix if re.match(r"GPU[0-9]+\t", line)}
    assert sorted(rows) == [0, 1, 2, 3]
    for source, destination in permutations(range(4), 2):
        cell = rows[source][1 + destination].strip()
        assert cell == "NV4"
        assert int(cell[2:]) == dump.nvlink_count(busid[source], busid[destination])
    assert all(rows[dev][1 + dev].strip() == "X" for dev in range(4))
    assert [int(rows[dev][numa_column]) for dev in range(4)] == [
        dump.numa_of(busid[dev]) for dev in range(4)
    ]

    pci_line = re.compile(r"([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]) [^:]+: (.+?) \(rev [0-9a-f]+\)")
    listed: dict[str, str] = {}
    for line in inventory_section("lspci nvidia/mellanox/cray/nic"):
        match = pci_line.fullmatch(line)
        assert match is not None, line
        if match.group(2) in LSPCI_IDS:
            listed[f"0000:{match.group(1)}"] = match.group(2)
    assert len(listed) == 8
    assert sorted(listed) == sorted(device.busid for device in dump.pci)
    for device in dump.pci:
        assert (device.vendor, device.device) == LSPCI_IDS[listed[device.busid]]
    assert sorted(bus for bus, name in listed.items() if "GA100" in name) == sorted(busid.values())

    status = inventory_section("nvidia-smi nvlink -s")
    links_per_gpu: dict[int, list[str]] = {}
    current = None
    for line in status:
        heading = re.match(r"GPU ([0-9]+): ", line)
        if heading:
            current = int(heading.group(1))
            links_per_gpu[current] = []
        elif line.strip():
            links_per_gpu[current].append(re.fullmatch(r"\s*Link [0-9]+: (.+)", line).group(1))
    assert sorted(links_per_gpu) == [0, 1, 2, 3]
    for dev, rates in links_per_gpu.items():
        assert rates == ["25 GB/s"] * 12
        pair_links = [row.count for row in dump.nvlinks if row.source_busid == busid[dev]]
        assert len(rates) == len(pair_links) * 4 == sum(pair_links) == 3 * 4
        assert len(rates) * 25 == 300


def mutate(change) -> str:
    root = ElementTree.fromstring(dump_text())
    change(root)
    return ElementTree.tostring(root, encoding="unicode")


def _second_nic_under_numa_three(root):
    cpu = root.find("cpu[@numaid='3']")
    extra = copy.deepcopy(cpu.find("pci[@busid='0000:01:00.0']"))
    extra.set("busid", "0000:02:00.0")
    net = extra.find("nic/net")
    net.set("name", "cxi4")
    net.set("dev", "4")
    cpu.append(extra)


def _no_nic_under_numa_three(root):
    cpu = root.find("cpu[@numaid='3']")
    cpu.remove(cpu.find("pci[@busid='0000:01:00.0']"))


REFUSALS = {
    "version 2": (lambda root: root.set("version", "2"), r"version '2'"),
    "gpu without rank": (
        lambda root: root.find("cpu/pci/gpu[@dev='0']").attrib.pop("rank"),
        r"<gpu> lacks required attribute rank",
    ),
    "asymmetric nvlink count 2 vs 4": (
        lambda root: root.find("cpu/pci/gpu[@dev='0']/nvlink[@target='0000:41:00.0']").set(
            "count", "2"),
        r"nvlink count 2 from 0000:03:00.0 to 0000:41:00.0 disagrees with 4",
    ),
    "two nics under one cpu": (
        _second_nic_under_numa_three, r"GPU dev 0 at 0000:03:00.0 has 2 NICs",
    ),
    "no nic under a cpu": (_no_nic_under_numa_three, r"GPU dev 0 at 0000:03:00.0 has 0 NICs"),
    "duplicate busid": (
        lambda root: root.find("cpu/pci[@busid='0000:01:00.0']").set("busid", "0000:03:00.0"),
        r"duplicate pci busid 0000:03:00.0",
    ),
    "nvlink target unknown busid": (
        lambda root: root.find("cpu/pci/gpu[@dev='0']/nvlink[@target='0000:82:00.0']").set(
            "target", "0000:99:00.0"),
        r"nvlink target 0000:99:00.0 is not a known gpu bus id",
    ),
    "net without speed": (
        lambda root: root.find("cpu/pci/nic/net[@name='cxi3']").attrib.pop("speed"),
        r"<net> lacks required attribute speed",
    ),
    "non-integer link_width": (
        lambda root: root.find("cpu/pci[@busid='0000:03:00.0']").set("link_width", "16.0"),
        r"link_width='16.0' is not a nonnegative integer",
    ),
}


@pytest.fixture
def no_schema_objects(monkeypatch):
    def refuse_build(*_args, **_kwargs):
        raise AssertionError("a schema object was built before the refusal")

    for name in SCHEMA_BUILDERS:
        monkeypatch.setattr(nccl_topology, name, refuse_build)


def test_n6_unmutated_serialization_is_a_valid_control():
    node, mesh = join(NcclTopologyDump.parse(mutate(lambda root: None)))
    assert (node, mesh) == join(load())


def test_n6_frozen_refusal_family_is_complete():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    assert sorted(REFUSALS) == sorted(frozen["cells"]["n6_refusals"])


@pytest.mark.parametrize("label", sorted(REFUSALS))
def test_n6_refusal_precedes_every_schema_object(label, no_schema_objects):
    change, message = REFUSALS[label]
    with pytest.raises(ValueError, match=message):
        join(NcclTopologyDump.parse(mutate(change)))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda root: root.find("cpu[@numaid='3']").append(ElementTree.Element("nvs")),
         r"element <nvs> is not accepted under <cpu>"),
        (lambda root: root.find("cpu/pci/gpu[@dev='1']").set("sm", "eighty"),
         r"attribute sm='eighty' is not a nonnegative integer"),
        (lambda root: root.find("cpu[@numaid='0']").set("host_hash", "0x1"),
         r"more than one host_hash"),
        (lambda root: root.find("cpu/pci/gpu[@dev='2']/nvlink[@target='0000:c1:00.0']").set(
            "tclass", "0x020000"),
         r"tclass 0x020000 disagrees"),
        (lambda root: root.find("cpu/pci/nic/net[@name='cxi3']").set("name", "CXI3"),
         r"net name 'CXI3' is not lowercase letters, digits and underscores"),
        (lambda root: root.find("cpu/pci/nic/net[@name='cxi3']").set("name", "cxi 3"),
         r"net name 'cxi 3' is not lowercase letters, digits and underscores"),
        (lambda root: root.find("cpu/pci/nic/net[@name='cxi0']").set("name", "CXI3"),
         r"duplicate net name \(case-insensitive\) cxi3"),
        (lambda root: root.find("cpu/pci[@busid='0000:03:00.0']").set("class", "0x030000"),
         r"pci 0000:03:00.0 class 0x030000 disagrees with its gpu child, which requires 0x030200"),
        (lambda root: root.find("cpu/pci[@busid='0000:01:00.0']").set("class", "0x020700"),
         r"pci 0000:01:00.0 class 0x020700 disagrees with its nic child, which requires 0x020000"),
        (lambda root: root.find("cpu/pci[@busid='0000:03:00.0']").set("vendor", "NVIDIA"),
         r"vendor='NVIDIA' is not lowercase hexadecimal"),
    ],
)
def test_reader_refuses_other_malformed_dumps(change, message, no_schema_objects):
    with pytest.raises(ValueError, match=message):
        NcclTopologyDump.parse(mutate(change))


def _partial_mesh(root):
    for dev, target in (("0", "0000:41:00.0"), ("1", "0000:03:00.0")):
        gpu = root.find(f"cpu/pci/gpu[@dev='{dev}']")
        gpu.remove(gpu.find(f"nvlink[@target='{target}']"))


def _shared_nic(root):
    moved = root.find("cpu[@numaid='2']/pci[@busid='0000:41:00.0']")
    root.find("cpu[@numaid='2']").remove(moved)
    root.find("cpu[@numaid='3']").append(moved)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_partial_mesh, r"GPU devs 0 and 1 share no NVLink"),
        (_shared_nic, r"NIC cxi3 at 0000:01:00.0 serves 2 GPUs"),
    ],
)
def test_join_refuses_unmodeled_shapes_before_building(change, message, no_schema_objects):
    with pytest.raises(ValueError, match=message):
        join(NcclTopologyDump.parse(mutate(change)))


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"global_rank_by_gpu_dev": {0: 0, 1: 1, 2: 2}}, ValueError),
        ({"global_rank_by_gpu_dev": {0: 0, 1: 0, 2: 2, 3: 3}}, ValueError),
        ({"global_rank_by_gpu_dev": {0: 0, 1: True, 2: 2, 3: 3}}, ValueError),
        ({"node_id": " "}, ValueError),
        ({"nvlink_link_rate_bps": 0}, ValueError),
        ({"nvlink_propagation_delay_ps": -1}, ValueError),
        ({"switch_input_buffer_bytes": 0}, ValueError),
    ],
)
def test_join_refuses_invalid_declared_inputs(overrides, error, no_schema_objects):
    with pytest.raises(error):
        join(load(), **overrides)


def test_n7_captured_fabric_manifest_round_trips(tmp_path):
    node, mesh = join(load())
    fabric = FabricTopologyManifest(nodes=[node], peer_fabrics=(mesh,), source="extracted")
    first = fabric.save(tmp_path / "fabric.json")
    loaded = FabricTopologyManifest.load(first)
    assert loaded == fabric
    second = loaded.save(tmp_path / "fabric-round-trip.json")
    assert first.read_bytes() == second.read_bytes()
    raw = fabric.to_dict()
    assert raw["source"] == "extracted"
    assert raw["physical_rendering_enabled"] is False
    (peer,) = raw["peer_fabrics"]
    assert peer["domain_id"] == "nid001056:nv4"
    assert (len(peer["ports"]), len(peer["links"]), len(peer["routes"])) == (48, 24, 12)


def _compatibility_payload(label: str, directory: Path) -> bytes:
    builders = {
        "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
        "m4_tp8": lambda: declared_manifest(tp=8),
        "rail_pp8": lambda: declared_pipeline_placement(8),
        "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
        "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
        "fabric_one_plus_one": lambda: disaggregated_manifests(
            prefill_nodes=1, decode_nodes=1).fabric,
        "fabric_one_plus_one_disabled": lambda: disaggregated_manifests(
            prefill_nodes=1, decode_nodes=1, render_physical_topology=False).fabric,
    }
    if label == "dgx_a100_peer_fabric_json":
        fabric = dgx_peer_fabric("a100", node_id="node-0", ranks=tuple(range(8)),
                                 propagation_delay_ps=1000, switch_input_buffer_bytes=65536)
        return json.dumps(asdict(fabric), sort_keys=True, separators=(",", ":")).encode()
    return builders[label]().save(directory / f"{label}.json").read_bytes()


@pytest.mark.parametrize(
    "label",
    sorted(json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["baseline"]["artifacts"]),
)
def test_compatibility_digest_is_unchanged(label, tmp_path):
    expected = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))["baseline"]["artifacts"]
    payload = _compatibility_payload(label, tmp_path)
    assert len(payload) == expected[label]["bytes"]
    assert hashlib.sha256(payload).hexdigest() == expected[label]["sha256"]


def test_freeze_pins_the_expectations_schema_and_capture_block():
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    assert frozen["schema"] == "simllm-nccl-topology-capture-expectations-v1"
    assert (frozen["task"], frozen["date"]) == ("PLACE-1", "2026-09-13")
    assert len(frozen["baseline"]["artifacts"]) == 8
    assert frozen["interface"] == {
        "module": "simllm.placement.nccl_topology",
        "reader": "NcclTopologyDump.load",
        "join": "captured_fabric_node",
        "peer_domain_id": "<node_id>:nv4",
        "nvlink_link_rate_bps_default": 200000000000,
        "fabric_source": "extracted",
        "peer_evidence_class": "declared",
    }
    capture = frozen["capture"]
    assert (capture["node"], capture["slurm_job"], capture["nccl_version"]) == (
        NODE, 58271200, "2.29.7",
    )
    assert capture["cpu"] == {"model": "AMD EPYC 7763", "numa_nodes": 4,
                              "numaid_file_order": [3, 0, 1, 2]}
    assert (capture["pci_devices"], capture["pcie"]) == (
        8, {"link_speed": "16.0 GT/s PCIe", "link_width": 16},
    )
    assert capture["gpus"] == {
        "count": 4, "vendor": "0x10de", "device": "0x20b0", "sm": 80,
        "busid_by_dev": {str(dev): busid for dev, busid in enumerate(GPU_BUS_IDS)},
        "numa_by_dev": {str(dev): numa for dev, numa in enumerate(GPU_NUMA)},
        "nvlink_rows": 12, "nvlink_count_per_pair": 4, "links_per_gpu": 12,
        "link_rate_gb_per_s": 25,
    }
    assert capture["nics"] == {
        "count": 4, "vendor": "0x17db", "device": "0x0501", "speed_mbit": 200000,
        "speed_bps": 200000000000,
        "busid_by_name": {name: busid for name, busid, _, _ in NICS_IN_NAME_ORDER},
        "numa_by_name": {name: numa for name, _, numa, _ in NICS_IN_NAME_ORDER},
        "gdr": 1, "maxconn": 128, "port": 1,
    }
    assert capture["affinity_gpu_dev_to_nic"] == {"0": "cxi3", "1": "cxi2", "2": "cxi1",
                                                  "3": "cxi0"}
    for name, digest in frozen["baseline"]["fixture"].items():
        if name != "directory":
            assert hashlib.sha256(fixture_bytes(name)).hexdigest() == digest, name
