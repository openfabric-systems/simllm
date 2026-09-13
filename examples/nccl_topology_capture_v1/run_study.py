"""Run the frozen PLACE-1 NCCL topology capture qualification.

The study answers one question: can the placement module read one real GPU
node's intra-node structure from NCCL's own topology dump, join it to the
fabric and peer-topology schemas, and drive the live peer packet path with
it, so that the captured node substitutes exactly for a hand-declared twin
without changing one byte of any existing manifest?

The input is the Perlmutter node `nid001056` fixture under
`tests/fixtures/nccl_topology/`. Cells N1, N2, N3, N4 and N7 are structural
exact guards over the reader and the join, N6 is a rejection control family,
and the eight compatibility digests are fatal by-construction identities.

Cell N5 is the only live and only scored cell. A dense tensor-parallel request
(one prefill step and two decode steps of the breakdown study's request, on a
small declared geometry so every NVLink packet can be simulated) runs through
`HtsimStepSink` on `rnic-nn-fluid` over `declared_manifest(tp=4,
gpus_per_node=4, hostname_pattern="nid001056")`, with a fixed 37,000 ps
compute provider and the pass-through direct-mesh profile of the live peer
packet study (25 GB/s per link). It runs once on the fabric the reader built
and once on a twin typed from the frozen literals of cells N2 and N3; the two
must agree on every `StepResult`, outcome and fabric document. Two relations
then hold on the captured run: every phase's packet service is at least its
peak endpoint bytes over the 100 GB/s of one pair's four links, and doubling
the declared per-link rate to 50 GB/s does not slow any phase or step.

Bulk artifacts go under the output root, which defaults to the study name
inside `SIMLLM_DATA_ROOT`; nothing machine specific reaches `results.json`.

Usage:

    python examples/nccl_topology_capture_v1/run_study.py
    python examples/nccl_topology_capture_v1/run_study.py --output-root <dir> --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, replace
from itertools import combinations, permutations
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
RESULTS_PATH = STUDY_DIR / "results.json"
STUDY_NAME = "nccl_topology_capture_v1"
EXPECTATIONS_COMMIT = "718e4566f44f99b11dfcb3183de00f66dadfba10"
RESULT_SCHEMA = "simllm-nccl-topology-capture-result-v1"
DATA_ROOT_ENV = "SIMLLM_DATA_ROOT"
FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "nccl_topology" / "perlmutter_a100_nid001056"

NODE_ID = "nid001056"
POOL_ROLE = "serving"
#: Declared NVLink propagation delay, the live peer packet study's value.
PROPAGATION_DELAY_PS = 1000
COMPUTE_PS = 37_000
TP_RANKS = (0, 1, 2, 3)
LINK_BYTES_PER_SECOND = 25_000_000_000
DOUBLED_LINK_BYTES_PER_SECOND = 50_000_000_000
PAIR_FLOOR_BYTES_PER_SECOND = 4 * LINK_BYTES_PER_SECOND
#: N5 dense geometry: one layer at hidden size 512, so the 2,048-token prefill
#: all-reduce carries 2,097,152 bytes and stays packet-simulable.
N5_DIMS = {
    "num_layers": 1,
    "hidden_size": 512,
    "intermediate_size": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_size": 64,
    "vocab_size": 49152,
    "dtype_bytes": 2,
}

#: Literals of expectations.md that the JSON registry does not repeat.
FROZEN_TEXT = {
    "host_hash": "0x94ecb6d613fd2af1",
    "arch": "x86_64",
    "cpu_vendor": "AuthenticAMD",
    "gpu_class": "0x030200",
    "nic_class": "0x020000",
    "gpu_gdr": 1,
    "nvlink_rows_per_gpu": 3,
    "nvlink_tclass": "0x030200",
    "nic_file_order": ["cxi3", "cxi0", "cxi1", "cxi2"],
}

#: The literal twin of cells N2 and N3, typed from expectations.md and never
#: derived from the reader: (dev, bus id, NUMA node, affine NIC) per GPU.
TWIN_GPUS = (
    (0, "0000:03:00.0", 3, "cxi3"),
    (1, "0000:41:00.0", 2, "cxi2"),
    (2, "0000:82:00.0", 1, "cxi1"),
    (3, "0000:c1:00.0", 0, "cxi0"),
)
#: (name, bus id, NUMA node, affine GPU rank) per NIC in name order.
TWIN_NICS = (
    ("cxi0", "0000:c2:00.0", 0, 3),
    ("cxi1", "0000:81:00.0", 1, 2),
    ("cxi2", "0000:42:00.0", 2, 1),
    ("cxi3", "0000:01:00.0", 3, 0),
)
TWIN_LANES_PER_PAIR = 4

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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True
    ).stdout


def _require_frozen_expectations() -> None:
    """Refuse to run unless the freeze precedes HEAD and is unchanged."""

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit("the PLACE-1 expectations commit is not an ancestor of HEAD")
    for name in ("expectations.md", "expectations.json"):
        relative = (STUDY_DIR / name).relative_to(REPOSITORY_ROOT).as_posix()
        if (STUDY_DIR / name).read_bytes() != _git_bytes(
            "show", f"{EXPECTATIONS_COMMIT}:{relative}"
        ):
            raise SystemExit(f"{name} differs from the frozen expectations commit")


def _default_output_root() -> Path | None:
    from simllm._local_config import path_from_env

    data_root = path_from_env(DATA_ROOT_ENV)
    return None if data_root is None else data_root / STUDY_NAME


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes().replace(b"\r\n", b"\n")


def _load_dump():
    from simllm.placement import NcclTopologyDump

    return NcclTopologyDump.load(FIXTURE_DIR / "nccl_topo.xml")


def _captured(dump, *, link_rate_bps: int = 8 * LINK_BYTES_PER_SECOND):
    from simllm.placement import FabricTopologyManifest, captured_fabric_node

    node, mesh = captured_fabric_node(
        dump,
        node_id=NODE_ID,
        pool_role=POOL_ROLE,
        global_rank_by_gpu_dev={dev: dev for dev in range(4)},
        nvlink_link_rate_bps=link_rate_bps,
        nvlink_propagation_delay_ps=PROPAGATION_DELAY_PS,
    )
    return FabricTopologyManifest(nodes=[node], peer_fabrics=(mesh,), source="extracted")


def _twin(*, link_rate_bps: int = 8 * LINK_BYTES_PER_SECOND):
    """The captured node rebuilt by hand from the frozen N2 and N3 literals."""

    from simllm.placement import (
        FabricLink,
        FabricNodePlacement,
        FabricTopologyManifest,
        GpuFabricPlacement,
        NicFabricPlacement,
        PeerFabric,
        PeerPortPlacement,
        PeerRoute,
    )

    gpus = tuple(
        GpuFabricPlacement(dev, busid, NODE_ID, f"numa-{numa}/pci-{busid}", f"{NODE_ID}:{nic}")
        for dev, busid, numa, nic in TWIN_GPUS
    )
    nics = tuple(
        NicFabricPlacement(f"{NODE_ID}:{name}", NODE_ID, f"numa-{numa}/pci-{busid}", rank)
        for name, busid, numa, rank in TWIN_NICS
    )
    domain = f"{NODE_ID}:nv4"
    ports, links = [], []
    for a, b in combinations(range(4), 2):
        for lane in range(TWIN_LANES_PER_PAIR):
            forward = f"{domain}:gpu-{a}:to-{b}:lane-{lane}"
            backward = f"{domain}:gpu-{b}:to-{a}:lane-{lane}"
            ports.extend((PeerPortPlacement(forward, gpu_rank=a),
                          PeerPortPlacement(backward, gpu_rank=b)))
            links.append(FabricLink(f"{domain}:link-{a}-{b}:lane-{lane}", forward, backward,
                                    link_rate_bps, PROPAGATION_DELAY_PS))
    routes = tuple(
        PeerRoute(source, destination, tuple(
            (f"{domain}:link-{min(source, destination)}-{max(source, destination)}:lane-{lane}",)
            for lane in range(TWIN_LANES_PER_PAIR)
        ))
        for source, destination in permutations(range(4), 2)
    )
    mesh = PeerFabric(domain, NODE_ID, tuple(ports), tuple(links), routes)
    node = FabricNodePlacement(NODE_ID, POOL_ROLE, gpus, nics)
    return FabricTopologyManifest(nodes=[node], peer_fabrics=(mesh,), source="extracted")


def run_n1(dump) -> dict[str, Any]:
    """Cell N1: every literal the parse must reproduce."""

    gpus = sorted(dump.gpus, key=lambda gpu: gpu.dev)
    gpu_busids = {gpu.busid for gpu in gpus}
    gpu_pci = [device for device in dump.pci if device.busid in gpu_busids]
    nic_pci = [device for device in dump.pci if device.busid not in gpu_busids]
    nvlink_rows = {
        gpu.dev: [row for row in dump.nvlinks if row.source_busid == gpu.busid] for gpu in gpus
    }
    return {
        "counts": {
            "cpus": len(dump.cpus),
            "gpus": len(dump.gpus),
            "nics": len(dump.nets),
            "nvlink_rows": len(dump.nvlinks),
            "pci": len(dump.pci),
        },
        "cpu": {
            "arch": sorted({cpu.arch for cpu in dump.cpus}),
            "host_hash": sorted({cpu.host_hash for cpu in dump.cpus}),
            "numaid_file_order": [cpu.numaid for cpu in dump.cpus],
            "vendor": sorted({cpu.vendor for cpu in dump.cpus}),
        },
        "pcie": {
            "link_speed": sorted({device.link_speed for device in dump.pci}),
            "link_width": sorted({device.link_width for device in dump.pci}),
        },
        "gpus": {
            "busid_by_dev": {str(gpu.dev): gpu.busid for gpu in gpus},
            "class": sorted({device.pci_class for device in gpu_pci}),
            "device": sorted({device.device for device in gpu_pci}),
            "gdr": sorted({gpu.gdr for gpu in gpus}),
            "numa_by_dev": {str(gpu.dev): dump.numa_of(gpu.busid) for gpu in gpus},
            "nvlink_count_per_pair": sorted({row.count for row in dump.nvlinks}),
            "nvlink_rows_per_gpu": sorted({len(rows) for rows in nvlink_rows.values()}),
            "nvlink_targets_are_the_other_gpus": all(
                sorted(row.target for row in nvlink_rows[gpu.dev])
                == sorted(gpu_busids - {gpu.busid})
                for gpu in gpus
            ),
            "nvlink_tclass": sorted({row.tclass for row in dump.nvlinks}),
            "rank_equals_dev": all(gpu.rank == gpu.dev for gpu in gpus),
            "sm": sorted({gpu.sm for gpu in gpus}),
            "vendor": sorted({device.vendor for device in gpu_pci}),
        },
        "nics": {
            "busid_by_name": {net.name: net.busid for net in dump.nets},
            "class": sorted({device.pci_class for device in nic_pci}),
            "device": sorted({device.device for device in nic_pci}),
            "file_order": [net.name for net in dump.nets],
            "gdr": sorted({net.gdr for net in dump.nets}),
            "maxconn": sorted({net.maxconn for net in dump.nets}),
            "numa_by_name": {net.name: dump.numa_of(net.busid) for net in dump.nets},
            "port": sorted({net.port for net in dump.nets}),
            "speed_bps": sorted({net.speed_bps for net in dump.nets}),
            "speed_mbit": sorted({net.speed for net in dump.nets}),
            "vendor": sorted({device.vendor for device in nic_pci}),
        },
        "affinity_gpu_dev_to_nic": {
            str(gpu.dev): next(
                net.name for net in dump.nets
                if dump.numa_of(net.busid) == dump.numa_of(gpu.busid)
            )
            for gpu in gpus
        },
    }


def run_n2(fabric) -> dict[str, Any]:
    """Cell N2: GPU and NIC placement of the captured node."""

    (node,) = fabric.nodes
    return {
        "gpus": [
            {
                "global_rank": gpu.global_rank,
                "gpu_id": gpu.gpu_id,
                "nic_id": gpu.nic_id,
                "node_id": gpu.node_id,
                "pcie_location": gpu.pcie_location,
            }
            for gpu in node.gpus
        ],
        "nics": [
            {
                "affine_gpu_rank": nic.affine_gpu_rank,
                "fabric_location": nic.fabric_location,
                "link_id": nic.link_id,
                "nic_id": nic.nic_id,
                "node_id": nic.node_id,
                "switch_id": nic.switch_id,
                "switch_port_id": nic.switch_port_id,
            }
            for nic in node.nics
        ],
        "node_id": node.node_id,
        "pool_role": node.pool_role,
    }


def run_n3(fabric) -> dict[str, Any]:
    """Cell N3: the direct NVLink mesh and its resolved paths."""

    (mesh,) = fabric.peer_fabrics
    pairs = {}
    for source, destination in permutations(sorted({r.source_rank for r in mesh.routes}), 2):
        rows = mesh.paths_between(source, destination)
        pairs[f"{source}-{destination}"] = {
            "direct": all(row.output_link is None for row in rows),
            "distinct_input_resources": len({row.input_resource for row in rows}),
            "paths": len(rows),
        }
    try:
        fabric.validate()
        valid = True
    except (TypeError, ValueError):
        valid = False
    return {
        "domain_id": mesh.domain_id,
        "evidence_class": mesh.evidence_class,
        "gpu_ports": len(mesh.ports),
        "link_propagation_delays_ps": sorted({link.propagation_delay_ps for link in mesh.links}),
        "link_rates_bps": sorted({link.link_rate_bps for link in mesh.links}),
        "links": len(mesh.links),
        "manifest_validates": valid,
        "pairs": pairs,
        "paths": sum(len(route.paths) for route in mesh.routes),
        "protocol": mesh.protocol,
        "routes": len(mesh.routes),
        "source": fabric.source,
        "switched": mesh.switched,
    }


def _inventory_section(title: str) -> list[str]:
    lines = _fixture_bytes("node_inventory.txt").decode("utf-8").splitlines()
    start = lines.index(f"--- {title} ---") + 1
    end = next(
        (index for index in range(start, len(lines))
         if lines[index].startswith("--- ") and lines[index].endswith(" ---")),
        len(lines),
    )
    return lines[start:end]


def run_n4(dump) -> dict[str, Any]:
    """Cell N4: the reader against the node inventory captured beside it."""

    busid = {gpu.dev: gpu.busid for gpu in dump.gpus}
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    matrix = [ansi.sub("", line) for line in _inventory_section("nvidia-smi topo -m")]
    numa_column = matrix[0].split("\t").index("NUMA Affinity")
    rows = {int(line.split("\t")[0][3:]): line.split("\t")
            for line in matrix if re.match(r"GPU[0-9]+\t", line)}
    nv4_all_pairs = sorted(rows) == [0, 1, 2, 3] and all(
        rows[source][1 + destination].strip() == "NV4"
        and dump.nvlink_count(busid[source], busid[destination]) == 4
        for source, destination in permutations(range(4), 2)
    )
    numa_column_matches = [int(rows[dev][numa_column]) for dev in range(4)] == [
        dump.numa_of(busid[dev]) for dev in range(4)
    ]

    pci_line = re.compile(r"([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]) [^:]+: (.+?) \(rev [0-9a-f]+\)")
    listed = {}
    for line in _inventory_section("lspci nvidia/mellanox/cray/nic"):
        match = pci_line.fullmatch(line)
        if match is not None and match.group(2) in LSPCI_IDS:
            listed[f"0000:{match.group(1)}"] = LSPCI_IDS[match.group(2)]
    lspci_matches = len(listed) == 8 and all(
        listed.get(device.busid) == (device.vendor, device.device) for device in dump.pci
    )

    links: dict[int, list[str]] = {}
    current = None
    for line in _inventory_section("nvidia-smi nvlink -s"):
        heading = re.match(r"GPU ([0-9]+): ", line)
        if heading:
            current = int(heading.group(1))
            links[current] = []
        elif line.strip() and current is not None:
            status = re.fullmatch(r"\s*Link [0-9]+: (.+)", line)
            links[current].append(status.group(1) if status else line)
    twelve_links = sorted(links) == [0, 1, 2, 3] and all(
        rates == ["25 GB/s"] * 12
        and len(rates) == 3 * 4
        == sum(row.count for row in dump.nvlinks if row.source_busid == busid[dev])
        for dev, rates in links.items()
    )
    return {
        "lspci_vendor_device": lspci_matches,
        "numa_affinity_column": numa_column_matches,
        "nv4_all_pairs": nv4_all_pairs,
        "nvlink_s_twelve_links_25gb": twelve_links,
    }


def _mutated_dump_text(change) -> str:
    root = ElementTree.fromstring(_fixture_bytes("nccl_topo.xml"))
    change(root)
    return ElementTree.tostring(root, encoding="unicode")


def _second_nic(root) -> None:
    cpu = root.find("cpu[@numaid='3']")
    extra = copy.deepcopy(cpu.find("pci[@busid='0000:01:00.0']"))
    extra.set("busid", "0000:02:00.0")
    extra.find("nic/net").set("name", "cxi4")
    extra.find("nic/net").set("dev", "4")
    cpu.append(extra)


def _no_nic(root) -> None:
    cpu = root.find("cpu[@numaid='3']")
    cpu.remove(cpu.find("pci[@busid='0000:01:00.0']"))


N6_MUTATIONS = {
    "version 2": lambda root: root.set("version", "2"),
    "gpu without rank": lambda root: root.find("cpu/pci/gpu[@dev='0']").attrib.pop("rank"),
    "asymmetric nvlink count 2 vs 4": lambda root: root.find(
        "cpu/pci/gpu[@dev='0']/nvlink[@target='0000:41:00.0']").set("count", "2"),
    "two nics under one cpu": _second_nic,
    "no nic under a cpu": _no_nic,
    "duplicate busid": lambda root: root.find("cpu/pci[@busid='0000:01:00.0']").set(
        "busid", "0000:03:00.0"),
    "nvlink target unknown busid": lambda root: root.find(
        "cpu/pci/gpu[@dev='0']/nvlink[@target='0000:82:00.0']").set("target", "0000:99:00.0"),
    "net without speed": lambda root: root.find("cpu/pci/nic/net[@name='cxi3']").attrib.pop(
        "speed"),
    "non-integer link_width": lambda root: root.find("cpu/pci[@busid='0000:03:00.0']").set(
        "link_width", "16.0"),
}


#: The refusal each N6 control must raise, matched as a message substring so the
#: harness scores the same rejection the tests do rather than any ValueError.
N6_EXPECTED_MESSAGES = {
    "version 2": "system version '2' is not '1'",
    "gpu without rank": "<gpu> lacks required attribute rank",
    "asymmetric nvlink count 2 vs 4": (
        "nvlink count 2 from 0000:03:00.0 to 0000:41:00.0 disagrees with 4"
    ),
    "two nics under one cpu": "GPU dev 0 at 0000:03:00.0 has 2 NICs",
    "no nic under a cpu": "GPU dev 0 at 0000:03:00.0 has 0 NICs",
    "duplicate busid": "duplicate pci busid 0000:03:00.0",
    "nvlink target unknown busid": "nvlink target 0000:99:00.0 is not a known gpu bus id",
    "net without speed": "<net> lacks required attribute speed",
    "non-integer link_width": "link_width='16.0' is not a nonnegative integer",
}


def run_n6(dump) -> dict[str, Any]:
    """Cell N6: each frozen refusal, raised before any schema object exists."""

    from simllm.placement import NcclTopologyDump, captured_fabric_node, nccl_topology

    def join(parsed):
        return captured_fabric_node(
            parsed, node_id=NODE_ID, pool_role=POOL_ROLE,
            global_rank_by_gpu_dev={dev: dev for dev in range(4)},
            nvlink_propagation_delay_ps=PROPAGATION_DELAY_PS,
        )

    control = join(NcclTopologyDump.parse(_mutated_dump_text(lambda root: None)))
    rows: dict[str, Any] = {
        "control_serialization_joins_identically": control == join(dump),
    }
    built: list[str] = []
    originals = {name: getattr(nccl_topology, name) for name in SCHEMA_BUILDERS}

    def sentinel(name):
        def build(*_args, **_kwargs):
            built.append(name)
            raise AssertionError(f"{name} built before the refusal")
        return build

    try:
        for name in SCHEMA_BUILDERS:
            setattr(nccl_topology, name, sentinel(name))
        for label, change in N6_MUTATIONS.items():
            built.clear()
            try:
                join(NcclTopologyDump.parse(_mutated_dump_text(change)))
            except ValueError as error:
                expected = N6_EXPECTED_MESSAGES[label]
                rows[label] = {
                    "expected_message": expected,
                    "message": str(error),
                    "refused": not built and expected in str(error),
                }
            except Exception as error:  # noqa: BLE001 - any other type is a wrong refusal
                rows[label] = {"message": f"{type(error).__name__}: {error}", "refused": False}
            else:
                rows[label] = {"message": None, "refused": False}
    finally:
        for name, value in originals.items():
            setattr(nccl_topology, name, value)
    return rows


def run_n7(fabric, output_root: Path) -> dict[str, Any]:
    """Cell N7: wire identity of the captured fabric manifest."""

    from simllm.placement import FabricTopologyManifest

    cell_root = output_root / "n7-round-trip"
    cell_root.mkdir(parents=True, exist_ok=True)
    first = fabric.save(cell_root / "fabric.json")
    loaded = FabricTopologyManifest.load(first)
    second = loaded.save(cell_root / "fabric-round-trip.json")
    raw = fabric.to_dict()
    peers = raw.get("peer_fabrics", [])
    return {
        "byte_identical_resave": first.read_bytes() == second.read_bytes(),
        "bytes": len(first.read_bytes()),
        "loaded_equals_built": loaded == fabric,
        "peer_inventory_in_to_dict": len(peers) == 1
        and (len(peers[0]["ports"]), len(peers[0]["links"]), len(peers[0]["routes"]))
        == (48, 24, 12),
        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
    }


def run_compatibility_digests(output_root: Path) -> dict[str, Any]:
    """The fatal identity: nothing built without the reader changes one byte."""

    from simllm.placement import (
        declared_manifest,
        declared_pipeline_placement,
        dgx_peer_fabric,
        disaggregated_manifests,
    )

    cell_root = output_root / "compatibility-digests"
    cell_root.mkdir(parents=True, exist_ok=True)
    savers = {
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
    rows: dict[str, Any] = {}
    for label, build in savers.items():
        payload = build().save(cell_root / f"{label}.json").read_bytes()
        rows[label] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    fabric = dgx_peer_fabric("a100", node_id="node-0", ranks=tuple(range(8)),
                             propagation_delay_ps=1000, switch_input_buffer_bytes=65536)
    payload = json.dumps(asdict(fabric), sort_keys=True, separators=(",", ":")).encode()
    (cell_root / "dgx_a100_peer_fabric_json.json").write_bytes(payload)
    rows["dgx_a100_peer_fabric_json"] = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return rows


def _run_request(workdir: Path, fabric, link_bytes_per_second: int) -> dict[str, Any]:
    """Drive the three-step dense request through the live peer packet path."""

    from examples.breakdown.run_breakdown import request_steps
    from examples.local_peer_packet_runtime_v1.run_study import FrozenCompute, profile_for
    from simllm.backends.peer_step import PeerPacketConfig
    from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ModelDims
    from simllm.core.step import step_record_to_json
    from simllm.core.step_io import step_result_to_json
    from simllm.placement import declared_manifest

    placement = declared_manifest(tp=4, gpus_per_node=4, hostname_pattern=NODE_ID)
    tp_ranks = tuple(placement.group_ranks(0, "tp"))
    (mesh,) = fabric.peer_fabrics
    config = HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=tp_ranks,
        dims=ModelDims(**N5_DIMS),
        workdir=workdir,
        placement_manifest=placement,
        provider=FrozenCompute(),
        peer_packet=PeerPacketConfig(
            fabric, ((mesh.domain_id, profile_for(False, rate=link_bytes_per_second)),)
        ),
    )
    sink = HtsimStepSink(config)
    steps, cursor = [], 0
    for record in request_steps()[:3]:
        record = replace(record, virtual_time_ps=cursor)
        result = sink(record)
        if result is None:
            raise SystemExit(f"step {record.step_index} produced no collective work")
        evidence = sink.peer_evidence[-1]
        phases = [
            {
                "nvlink_bytes": artifact.local_phase.phase.nvlink_bytes,
                "operation_id": artifact.local_phase.phase.phase.operation_id,
                "peak_endpoint_bytes": artifact.local_phase.phase.nvlink_peak_endpoint_bytes,
                "service_ps": artifact.local_phase.service_ps,
            }
            for artifact in evidence.artifacts
            if artifact.local_phase is not None
        ]
        steps.append({
            "locality": asdict(sink.locality_outcomes[-1]),
            "outcome": asdict(sink.outcomes[-1]),
            "phases": phases,
            "record": step_record_to_json(record),
            "result": step_result_to_json(result),
        })
        cursor = result.completed_at_ps
    final = sink.close_peer_packets()
    return {
        "fabric": fabric.to_dict(),
        "quiescent_after_close": all(not row["has_pending_physical_work"] for row in final),
        "steps": steps,
        "tp_ranks": list(tp_ranks),
    }


def run_n5(dump, output_root: Path) -> dict[str, Any]:
    """Cell N5: captured and literal-twin fabrics drive the same live request."""

    captured = _run_request(output_root / "n5-captured", _captured(dump), LINK_BYTES_PER_SECOND)
    twin = _run_request(output_root / "n5-twin", _twin(), LINK_BYTES_PER_SECOND)
    doubled = _run_request(
        output_root / "n5-doubled-rate",
        _captured(dump, link_rate_bps=8 * DOUBLED_LINK_BYTES_PER_SECOND),
        DOUBLED_LINK_BYTES_PER_SECOND,
    )
    for label, run in (("captured", captured), ("twin", twin), ("doubled-rate", doubled)):
        _write_json(output_root / f"n5-{label}.json", run)

    floor_rows = []
    for step in captured["steps"]:
        for phase in step["phases"]:
            floor_ps = -(-phase["peak_endpoint_bytes"] * 10**12 // PAIR_FLOOR_BYTES_PER_SECOND)
            floor_rows.append({"floor_ps": floor_ps, "service_ps": phase["service_ps"]})
    never_slower = all(
        fast["result"]["step_latency_ps"] <= base["result"]["step_latency_ps"]
        and len(fast["phases"]) == len(base["phases"])
        and all(quick["service_ps"] <= slow["service_ps"]
                for quick, slow in zip(fast["phases"], base["phases"], strict=True))
        for fast, base in zip(doubled["steps"], captured["steps"], strict=True)
    )

    def column(run, key):
        return [step[key] for step in run["steps"]]

    return {
        "compute_ps": COMPUTE_PS,
        "dims": N5_DIMS,
        "link_bytes_per_second": LINK_BYTES_PER_SECOND,
        "pair_floor": {
            "bytes_per_second": PAIR_FLOOR_BYTES_PER_SECOND,
            "held": bool(floor_rows) and all(
                row["service_ps"] >= row["floor_ps"] for row in floor_rows
            ),
            "instances": len(floor_rows),
            "minimum_slack_ps": min(row["service_ps"] - row["floor_ps"] for row in floor_rows),
        },
        "profile": "rnic-nn-fluid",
        "propagation_delay_ps": PROPAGATION_DELAY_PS,
        "rate_doubling": {
            "doubled_link_bytes_per_second": DOUBLED_LINK_BYTES_PER_SECOND,
            "never_slower": never_slower,
            "phase_services_ps": [[phase["service_ps"] for phase in step["phases"]]
                                  for step in doubled["steps"]],
            "step_latency_ps": [step["result"]["step_latency_ps"] for step in doubled["steps"]],
        },
        "steps": [
            {
                "backend_runs": step["locality"]["backend_runs"],
                "fabric_directed_bytes": step["locality"]["fabric_directed_bytes"],
                "nvlink_directed_bytes": step["locality"]["nvlink_directed_bytes"],
                "phases": [
                    {**phase, "floor_ps": -(-phase["peak_endpoint_bytes"] * 10**12
                                            // PAIR_FLOOR_BYTES_PER_SECOND)}
                    for phase in step["phases"]
                ],
                "result": step["result"],
                "scheduled": step["record"]["scheduled"],
                "step_index": step["record"]["step_index"],
            }
            for step in captured["steps"]
        ],
        "tp_ranks": captured["tp_ranks"],
        "twin_identity": {
            "fabric_to_dict": captured["fabric"] == twin["fabric"],
            "locality_outcomes": column(captured, "locality") == column(twin, "locality"),
            "outcomes": column(captured, "outcome") == column(twin, "outcome"),
            "phase_services": column(captured, "phases") == column(twin, "phases"),
            "step_results": column(captured, "result") == column(twin, "result"),
        },
        "quiescent_after_close": all(
            run["quiescent_after_close"] for run in (captured, twin, doubled)
        ),
    }


def analyze_observation(observation: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Apply every frozen guard, then score the N5 identity and its two relations."""

    findings: list[str] = []
    cells = observation["cells"]
    frozen_cells = frozen["cells"]
    capture = frozen["capture"]

    if observation.get("expectations_commit") != EXPECTATIONS_COMMIT:
        findings.append("expectations commit identity")
    for name, digest in frozen["baseline"]["fixture"].items():
        if name != "directory" and observation["fixture_sha256"].get(name) != digest:
            findings.append(f"fixture digest {name}")

    # N1
    n1 = cells["n1_parse"]
    if n1["counts"] != frozen_cells["n1_parse"]:
        findings.append("n1: element counts")
    expected_cpu = {
        "arch": [FROZEN_TEXT["arch"]],
        "host_hash": [FROZEN_TEXT["host_hash"]],
        "numaid_file_order": capture["cpu"]["numaid_file_order"],
        "vendor": [FROZEN_TEXT["cpu_vendor"]],
    }
    if n1["cpu"] != expected_cpu:
        findings.append("n1: cpu rows")
    if n1["pcie"] != {"link_speed": [capture["pcie"]["link_speed"]],
                      "link_width": [capture["pcie"]["link_width"]]}:
        findings.append("n1: PCIe link")
    frozen_gpus, frozen_nics = capture["gpus"], capture["nics"]
    expected_gpus = {
        "busid_by_dev": frozen_gpus["busid_by_dev"],
        "class": [FROZEN_TEXT["gpu_class"]],
        "device": [frozen_gpus["device"]],
        "gdr": [FROZEN_TEXT["gpu_gdr"]],
        "numa_by_dev": frozen_gpus["numa_by_dev"],
        "nvlink_count_per_pair": [frozen_gpus["nvlink_count_per_pair"]],
        "nvlink_rows_per_gpu": [FROZEN_TEXT["nvlink_rows_per_gpu"]],
        "nvlink_targets_are_the_other_gpus": True,
        "nvlink_tclass": [FROZEN_TEXT["nvlink_tclass"]],
        "rank_equals_dev": True,
        "sm": [frozen_gpus["sm"]],
        "vendor": [frozen_gpus["vendor"]],
    }
    for key, expected in expected_gpus.items():
        if n1["gpus"].get(key) != expected:
            findings.append(f"n1: gpu {key}")
    expected_nics = {
        "busid_by_name": frozen_nics["busid_by_name"],
        "class": [FROZEN_TEXT["nic_class"]],
        "device": [frozen_nics["device"]],
        "file_order": FROZEN_TEXT["nic_file_order"],
        "gdr": [frozen_nics["gdr"]],
        "maxconn": [frozen_nics["maxconn"]],
        "numa_by_name": frozen_nics["numa_by_name"],
        "port": [frozen_nics["port"]],
        "speed_bps": [frozen_nics["speed_bps"]],
        "speed_mbit": [frozen_nics["speed_mbit"]],
        "vendor": [frozen_nics["vendor"]],
    }
    for key, expected in expected_nics.items():
        if n1["nics"].get(key) != expected:
            findings.append(f"n1: nic {key}")
    if n1["affinity_gpu_dev_to_nic"] != capture["affinity_gpu_dev_to_nic"]:
        findings.append("n1: NUMA affinity")

    # N2
    frozen_n2 = frozen_cells["n2_join"]
    n2 = cells["n2_join"]
    if (n2["node_id"], n2["pool_role"]) != (frozen_n2["node_id"], frozen_n2["pool_role"]):
        findings.append("n2: node identity")
    expected_gpu_rows = [
        {
            "global_rank": int(dev),
            "gpu_id": busid,
            "nic_id": frozen_n2["gpu_nic_by_rank"][dev],
            "node_id": frozen_n2["node_id"],
            "pcie_location": f"numa-{frozen_gpus['numa_by_dev'][dev]}/pci-{busid}",
        }
        for dev, busid in sorted(frozen_gpus["busid_by_dev"].items(), key=lambda row: int(row[0]))
    ]
    if n2["gpus"] != expected_gpu_rows:
        findings.append("n2: GPU placement rows")
    expected_nic_rows = [
        {
            "affine_gpu_rank": rank,
            "fabric_location": (
                f"numa-{frozen_nics['numa_by_name'][name]}/pci-{frozen_nics['busid_by_name'][name]}"
            ),
            "link_id": None,
            "nic_id": f"{frozen_n2['node_id']}:{name}",
            "node_id": frozen_n2["node_id"],
            "switch_id": None,
            "switch_port_id": None,
        }
        for name, rank in zip(
            frozen_n2["nic_order"], frozen_n2["affine_gpu_rank_in_nic_order"], strict=True
        )
    ]
    if n2["nics"] != expected_nic_rows:
        findings.append("n2: NIC placement rows")

    # N3
    frozen_n3 = frozen_cells["n3_mesh"]
    n3 = cells["n3_mesh"]
    for key in ("links", "gpu_ports", "routes", "paths"):
        if n3[key] != frozen_n3[key]:
            findings.append(f"n3: {key}")
    if n3["link_rates_bps"] != [frozen_n3["link_rate_bps"]]:
        findings.append("n3: link rate")
    if n3["link_propagation_delays_ps"] != [PROPAGATION_DELAY_PS]:
        findings.append("n3: propagation delay")
    if len(n3["pairs"]) != frozen_n3["routes"] or any(
        row != {"direct": True, "distinct_input_resources": frozen_n3["paths_per_route"],
                "paths": frozen_n3["paths_per_route"]}
        for row in n3["pairs"].values()
    ):
        findings.append("n3: resolved paths per ordered pair")
    interface = frozen["interface"]
    if (n3["domain_id"], n3["evidence_class"], n3["source"], n3["protocol"], n3["switched"]) != (
        interface["peer_domain_id"].replace("<node_id>", NODE_ID),
        interface["peer_evidence_class"], interface["fabric_source"], "nvlink", False,
    ):
        findings.append("n3: domain identity and provenance")
    if n3["manifest_validates"] is not True:
        findings.append("n3: fabric manifest validation")

    # N4
    for check in frozen_cells["n4_cross_check"]:
        if cells["n4_cross_check"].get(check) is not True:
            findings.append(f"n4: {check}")

    # N5, the scored rows
    frozen_n5 = frozen_cells["n5_live"]
    n5 = cells["n5_live"]
    if (n5["profile"], n5["tp_ranks"], n5["compute_ps"]) != (
        frozen_n5["profile"], frozen_n5["tp_ranks"], frozen_n5["compute_ps"],
    ):
        findings.append("n5: live configuration")
    if n5["pair_floor"]["bytes_per_second"] != frozen_n5["pair_floor_bytes_per_second"]:
        findings.append("n5: pair floor rate")
    identity_names = {
        "StepResult": "step_results", "outcomes": "outcomes", "fabric_to_dict": "fabric_to_dict",
    }
    identity_held = all(
        n5["twin_identity"][identity_names[name]] is True for name in frozen_n5["twin_identity"]
    ) and n5["twin_identity"]["locality_outcomes"] is True
    if not identity_held:
        findings.append("n5: captured and literal twin runs disagree")
    for step in n5["steps"]:
        label = f"n5 step {step['step_index']}"
        if step["fabric_directed_bytes"] != frozen_n5["fabric_bytes"]:
            findings.append(f"{label}: fabric bytes")
        if not step["nvlink_directed_bytes"] > 0:
            findings.append(f"{label}: NVLink bytes")
        if step["backend_runs"] != 0:
            findings.append(f"{label}: fabric backend invoked")
        if not step["phases"]:
            findings.append(f"{label}: no executed local phase")
    if n5["quiescent_after_close"] is not True:
        findings.append("n5: packet calendars not quiescent after close")
    relations = {
        "pair_floor": n5["pair_floor"]["held"] is True,
        "rate_doubling_never_slower": n5["rate_doubling"]["never_slower"] is True,
    }
    for name, held in relations.items():
        if not held:
            findings.append(f"n5: relation {name}")

    # N6
    n6 = cells["n6_refusals"]
    if n6.get("control_serialization_joins_identically") is not True:
        findings.append("n6: unmutated control")
    for refusal in frozen_cells["n6_refusals"]:
        if n6.get(refusal, {}).get("refused") is not True:
            findings.append(f"n6: {refusal} was not refused before any schema object")

    # N7
    n7 = cells["n7_round_trip"]
    if frozen_cells["n7_round_trip"] is True and not all(
        n7[key] is True
        for key in ("byte_identical_resave", "loaded_equals_built", "peer_inventory_in_to_dict")
    ):
        findings.append("n7: wire identity")

    # The fatal compatibility digests
    for label, expected in frozen["baseline"]["artifacts"].items():
        row = cells["compatibility_digests"].get(label, {})
        if row.get("bytes") != expected["bytes"]:
            findings.append(f"digest {label}: bytes")
        if row.get("sha256") != expected["sha256"]:
            findings.append(f"digest {label}: sha256")

    evidence = frozen["evidence"]
    return {
        "evidence": {
            "fatal_compatibility_digests": len(frozen["baseline"]["artifacts"]),
            "rejection_control_family": {
                "controls": len(frozen_cells["n6_refusals"]),
                "refused": sum(
                    n6.get(label, {}).get("refused") is True
                    for label in frozen_cells["n6_refusals"]
                ),
            },
            "scored_identity_family": {
                "families": evidence["scored_identity_family"],
                "held": int(identity_held),
            },
            "scored_relation_instances": {
                "held": sum(relations.values()),
                "instances": evidence["scored_relation_instances"],
            },
            "structural_guard_cells": len(evidence["structural_guard_cells"]),
        },
        "findings": findings,
        "status": "PASS" if not findings else "VOID",
    }


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute every cell and return the complete tracked summary."""

    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    dump = _load_dump()
    captured = _captured(dump)
    cells = {
        "compatibility_digests": run_compatibility_digests(output_root),
        "n1_parse": run_n1(dump),
        "n2_join": run_n2(captured),
        "n3_mesh": run_n3(captured),
        "n4_cross_check": run_n4(dump),
        "n5_live": run_n5(dump, output_root),
        "n6_refusals": run_n6(dump),
        "n7_round_trip": run_n7(captured, output_root),
    }
    fixture_sha256 = {
        name: hashlib.sha256(_fixture_bytes(name)).hexdigest()
        for name in sorted(frozen["baseline"]["fixture"])
        if name != "directory"
    }
    observation = {
        "cells": cells,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "fixture_sha256": fixture_sha256,
    }
    analysis = analyze_observation(observation, frozen)
    summary = {
        "cells": cells,
        "evidence": analysis["evidence"],
        "expectations_commit": EXPECTATIONS_COMMIT,
        "findings": analysis["findings"],
        "fixture_sha256": fixture_sha256,
        "implementation_commit": _git_bytes("rev-parse", "HEAD").decode().strip(),
        "schema": RESULT_SCHEMA,
        "status": analysis["status"],
    }
    text = json.dumps(summary, sort_keys=True)
    for machine_path in {str(output_root), str(REPOSITORY_ROOT), str(Path.home())}:
        if machine_path in text:
            summary["findings"].append("a machine path reached the tracked summary")
            summary["status"] = "VOID"
            break
    return summary


def comparable(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop the one key that legitimately moves between reproducing runs."""

    return {key: value for key, value in summary.items() if key != "implementation_commit"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PLACE-1 NCCL topology capture study")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help=(
            "external artifact directory, fresh for each run; defaults to the "
            f"study name inside {DATA_ROOT_ENV}"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare this run against the tracked results.json instead of writing it",
    )
    args = parser.parse_args()
    if args.output_root is None:
        raise SystemExit(f"provide --output-root or set {DATA_ROOT_ENV}")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    _require_frozen_expectations()

    summary = run_study(output_root)
    _write_json(output_root / "results.json", summary)
    print(json.dumps(
        {key: summary[key] for key in ("evidence", "findings", "status")},
        indent=2, sort_keys=True,
    ))
    n5 = summary["cells"]["n5_live"]
    for step in n5["steps"]:
        prefill = max(step["phases"], key=lambda phase: phase["peak_endpoint_bytes"])
        print(
            f"N5 step {step['step_index']} latency_ps={step['result']['step_latency_ps']} "
            f"phases={len(step['phases'])} peak_endpoint_bytes={prefill['peak_endpoint_bytes']} "
            f"service_ps={prefill['service_ps']} floor_ps={prefill['floor_ps']}"
        )
    print(f"N5 twin identity {n5['twin_identity']}")
    print(f"N5 doubled-rate step latency_ps {n5['rate_doubling']['step_latency_ps']}")

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
