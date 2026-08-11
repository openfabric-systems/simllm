"""Dry-run registry for the frozen BACK-28 reader-parity study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SIMLLM_BASE_COMMIT = "9923c9f0add6b6f23a0019382962931e1792bc47"
EXPECTATIONS_COMMIT = "f50ed49581f072eb50d4c3e69445217cb2877c36"
EFFECTIVE_HASHES = {
    "v2": "4a94c6ec23c0af9a18524d33dbb3127dd1d4cde4dcfced7e972fdb1dda5dfebf",
    "v3_host": "4ffabebbb9c6f5aace706f241af030b95c286c607a6e4f9e39a146e5065dfa17",
    "v3_proxy": "a9cc2fa1df269f75d6c8d48ba27bff81dd5c65f086ea9c4fe3f3eaadb264cee3",
    "v3_gpu": "cd4da0c3635006ce3a02f1a19b1ecd1700ca92fa6758ee87a77bccca6f15c4ad",
}
V1_EFFECTIVE_HASH = "a9732c130d2ed0075668c7ee1f77c742492ca059f1c50b1ca35c078799deaa9c"
V1_CONFIG_HASH = "69f20997fede3a9a00b386a5a0412f948dba5bc1b6eb0c7e93d6d6dd85e01d0c"
BYPASS_CONFIG_HASH = "c750be3ba90023987478e6ecd111ee70ad90c02f669470e547ef252e047afc2b"


def _disabled_penalties() -> dict[str, object]:
    return {
        name: {"kind": "disabled"}
        for name in ("acs", "ddio_miss", "gpu_direct", "iommu", "numa", "switch_path")
    }


def _credits() -> dict[str, int]:
    return {
        "completion_data_credits": 4096,
        "completion_header_credits": 64,
        "nonposted_header_credits": 64,
        "posted_data_credits": 4096,
        "posted_header_credits": 64,
    }


def _fabric() -> dict[str, object]:
    return {
        "analytical_seed": 0,
        "completion_buffer_bytes": 65536,
        "completion_buffer_release_latency_ps": 0,
        "completion_overhead_bytes": 20,
        "credit_return_latency_ps": 0,
        "data_credit_unit_bytes": 16,
        "device_to_host_credits": _credits(),
        "generation": 5,
        "host_store_latency_ps": [0],
        "host_to_device_credits": _credits(),
        "lane_count": 16,
        "max_outstanding_read_requests": 64,
        "max_payload_size_bytes": 256,
        "max_read_request_size_bytes": 512,
        "max_tlps_per_transaction": 1_000_000,
        "paths": [
            {
                "analytical_penalties": _disabled_penalties(),
                "base_latency_ps": 0,
                "enabled": True,
                "endpoint": endpoint,
                "path_id": path_id,
            }
            for path_id, endpoint in (
                (1, "mmio_bar"),
                (2, "host_pinned_memory"),
                (3, "gpu_memory"),
            )
        ],
        "posted_write_overhead_bytes": 24,
        "posted_write_visibility_latency_ps": [0],
        "read_completion_boundary_bytes": 64,
        "read_completion_latency_ps": [0],
        "read_request_overhead_bytes": 24,
    }


def _allocation(
    allocation_id: int,
    object_kind: str,
    owner_kind: str,
    owner_id: int,
    endpoint: str,
    path_id: int,
    slot: int,
    page_count: int,
    length_bytes: int,
    *,
    mkey: int | None = None,
) -> dict[str, object]:
    page_size = 4096
    result: dict[str, object] = {
        "allocation_id": allocation_id,
        "device_owner_id": 920,
        "endpoint": endpoint,
        "length_bytes": length_bytes,
        "object_kind": object_kind,
        "owner_id": owner_id,
        "owner_kind": owner_kind,
        "pages": {
            "page_size_bytes": page_size,
            "physical_page_addresses": [
                0x400000000 + (slot * 4 + page) * page_size
                for page in range(page_count)
            ],
        },
        "path_id": path_id,
        "virtual_address": 0x300000000 + slot * 4 * page_size,
    }
    if mkey is not None:
        result["mkey"] = mkey
    return result


def _effective_v3(shape: str) -> dict[str, object]:
    gpu = shape == "gpu_initiated"
    proxy = shape == "cpu_proxy"
    ring_endpoint = "gpu_memory" if gpu else "host_pinned_memory"
    ring_path = 3 if gpu else 2
    data_endpoint = "host_pinned_memory" if shape == "host_cpu_driver" else "gpu_memory"
    data_path = 2 if data_endpoint == "host_pinned_memory" else 3
    allocations = [
        _allocation(21, "qpc_icm", "queue_pair", 19, "host_pinned_memory", 2, 1, 1, 256),
        _allocation(22, "sq_ring", "send_queue", 201, ring_endpoint, ring_path, 2, 1, 1024),
        _allocation(23, "rq_ring", "receive_queue", 202, "host_pinned_memory", 2, 3, 1, 1024),
        _allocation(24, "cq_ring", "completion_queue", 203, ring_endpoint, ring_path, 4, 1, 1024),
        _allocation(25, "doorbell_record", "send_queue", 201, ring_endpoint, ring_path, 5, 1, 4),
        _allocation(26, "data_region", "memory_region", 177, data_endpoint, data_path, 6, 2, 8192, mkey=177),
    ]
    if proxy:
        allocations.append(
            _allocation(
                27,
                "descriptor_queue",
                "submission_producer",
                7202,
                "host_pinned_memory",
                2,
                7,
                1,
                4096,
            )
        )
    shape_fields = {
        "host_cpu_driver": {
            "cq_consumer_id": 8101,
            "cq_consumer_kind": "host_cpu_driver",
            "descriptor_queue_allocation_id": 0,
            "descriptor_queue_endpoint": "none",
            "descriptor_writer_id": 0,
            "descriptor_writer_kind": "none",
            "producer_id": 7101,
            "producer_kind": "host_cpu_driver",
            "uar_mapping_owner": "host_cpu",
        },
        "cpu_proxy": {
            "cq_consumer_id": 8102,
            "cq_consumer_kind": "cpu_proxy",
            "descriptor_queue_allocation_id": 27,
            "descriptor_queue_endpoint": "host_pinned_memory",
            "descriptor_writer_id": 7202,
            "descriptor_writer_kind": "gpu",
            "producer_id": 7102,
            "producer_kind": "cpu_proxy",
            "uar_mapping_owner": "host_cpu",
        },
        "gpu_initiated": {
            "cq_consumer_id": 8103,
            "cq_consumer_kind": "gpu",
            "descriptor_queue_allocation_id": 0,
            "descriptor_queue_endpoint": "none",
            "descriptor_writer_id": 0,
            "descriptor_writer_kind": "none",
            "producer_id": 7103,
            "producer_kind": "gpu",
            "uar_mapping_owner": "gpu",
        },
    }[shape]
    submission = {
        **shape_fields,
        "producer_shape": shape,
        "queue_endpoint": ring_endpoint,
        "rnic_requester_id": 9100,
    }
    return {
        "dma": {
            "enabled": True,
            "fabric": _fabric(),
            "fabric_scope": "owned",
            "work_queue": {
                "pcie_completion_ordering_domain": 406,
                "pcie_cq_first_byte_offset": 0,
                "pcie_cq_memory_path_id": ring_path,
                "pcie_cqe_bytes": 64,
                "pcie_doorbell_record_bytes": 4,
                "pcie_doorbell_record_first_byte_offset": 0,
                "pcie_doorbell_record_path_id": ring_path,
                "pcie_sq_first_byte_offset": 0,
                "pcie_sq_memory_path_id": ring_path,
                "pcie_submission_ordering_domain": 403,
                "pcie_uar_doorbell_bytes": 8,
                "pcie_uar_first_byte_offset": 0,
                "pcie_uar_path_id": 1,
                "pcie_wqe_bytes": 64,
            },
        },
        "host_memory": {
            "allocations": allocations,
            "device_owner_id": 920,
            "enabled": True,
            "registry": {
                "mpt_entry_bytes": 64,
                "mpt_first_byte_offset": 0,
                "mtt_entry_bytes": 8,
                "mtt_first_byte_offset": 0,
                "queue_page_list_entry_bytes": 8,
                "queue_page_list_first_byte_offset": 0,
                "translation_path_id": 2,
            },
            "work_queue": {
                "cq_ring_allocation_id": 24,
                "doorbell_record_allocation_id": 25,
                "qpc_context_bytes": 256,
                "qpc_icm_allocation_id": 21,
                "rq_ring_allocation_id": 23,
                "sq_ring_allocation_id": 22,
            },
        },
        "network": {"enabled": False},
        "qpc": {"enabled": True},
        "schema": "simllm-rnic-effective-hardware-v3",
        "submission": submission,
        "work_queue": {
            "cq_depth": 16,
            "qpc_lookup_service_ps": 0,
            "scheduler_service_ps": 0,
            "sq_depth": 16,
        },
    }


def _effective_v2() -> dict[str, object]:
    value = _effective_v3("host_cpu_driver")
    value.pop("submission")
    value["schema"] = "simllm-rnic-effective-hardware-v2"
    return value


def _effective_v1() -> dict[str, object]:
    return {
        "dma": {"enabled": False},
        "network": {"enabled": True},
        "qpc": {"enabled": True},
        "schema": "simllm-rnic-effective-hardware-v1",
        "work_queue": {
            "cq_depth": 64,
            "cqe_write_service_ps": 13,
            "doorbell_service_ps": 37,
            "qpc_lookup_service_ps": 5,
            "scheduler_service_ps": 7,
            "sq_depth": 64,
            "wqe_fetch_service_ps": 11,
        },
    }


def _config(effective: dict[str, object] | None, *, bypass: bool = False) -> dict[str, object]:
    return {
        "authority": "AtlahsWqeLedger" if bypass else "SimllmNativeRnicSession",
        "effective_hardware": effective,
        "hardware_config_sha256": None if bypass else _digest(effective),
        "hardware_mode": "bypass" if bypass else "structural",
        "schema": "simllm-rnic-session-config-v1",
        "session_id": "session-bypass" if bypass else "session-native",
        "transport_policy": "rnic-nn-fluid" if bypass else "rnic-nn",
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


MUTATIONS = (
    ("N001", "v2", "unsupported schema", "1088-1100"),
    ("N002", "v2", "v2 missing host_memory", "1106-1110"),
    ("N003", "v2", "v2 carries submission", "1106-1110"),
    ("N004", "v3_host", "v3 missing submission", "1101-1105"),
    ("N005", "v2", "root has an extra field", "391-400,1106-1110"),
    ("N006", "v2", "network has an extra field", "751-758"),
    ("N007", "v2", "qpc enabled is not boolean", "751-758"),
    ("N008", "v2", "host memory with DMA disabled", "1129-1155"),
    ("N009", "v2", "host memory with QPC disabled", "1118-1155"),
    ("N010", "v2", "DMA has an extra field", "1134-1150"),
    ("N011", "v2", "DMA fabric scope is unknown", "1134-1141"),
    ("N012", "v2", "fabric field set is incomplete", "579-594"),
    ("N013", "v2", "PCIe generation exceeds five", "597-601"),
    ("N014", "v2", "lane count is not a power of two", "602-605"),
    ("N015", "v2", "MPS is below 128 bytes", "607-622"),
    ("N016", "v2", "read completion boundary is invalid", "624-630"),
    ("N017", "v2", "positive fabric field is zero", "631-637"),
    ("N018", "v2", "data credit unit is not a power of two", "638-645"),
    ("N019", "v2", "credit count is zero", "536-557"),
    ("N020", "v2", "credit count exceeds uint32", "551-557"),
    ("N021", "v2", "data credits cannot hold one MPS", "558-563"),
    ("N022", "v2", "latency profile has two samples", "567-577"),
    ("N023", "v2", "PCIe path list is empty", "670-674"),
    ("N024", "v2", "PCIe path IDs are not ascending", "675-689"),
    ("N025", "v2", "PCIe path ID is zero", "504-513"),
    ("N026", "v2", "PCIe path ID exceeds uint32", "508-513"),
    ("N027", "v2", "enabled PCIe path misses endpoint", "518-522"),
    ("N028", "v2", "PCIe path endpoint is unknown", "527-532"),
    ("N029", "v2", "PCIe path latency exceeds int64", "523-526"),
    ("N030", "v2", "disabled PCIe path keeps active fields", "514-516"),
    ("N031", "v2", "DMA binding field set is incomplete", "694-709"),
    ("N032", "v2", "positive DMA binding field is zero", "710-718"),
    ("N033", "v2", "DMA binding offset reaches 4096", "719-726"),
    ("N034", "v2", "DMA binding path exceeds uint32", "728-740"),
    ("N035", "v2", "DMA binding path is missing", "741-747"),
    ("N036", "v2", "DMA binding path endpoint is wrong", "728-747"),
    ("N037", "v3_host", "submission adds a second CQ consumer field", "767-776"),
    ("N038", "v3_host", "producer identity is zero", "777-785"),
    ("N039", "v3_host", "CQ consumer identity exceeds uint32", "777-785"),
    ("N040", "v3_host", "RNIC requester identity is zero", "777-785"),
    ("N041", "v3_host", "producer shape is unknown", "814-849"),
    ("N042", "v3_host", "host producer kind disagrees", "814-824"),
    ("N043", "v3_host", "host descriptor writer is nonzero", "814-824"),
    ("N044", "v3_proxy", "proxy descriptor writer kind disagrees", "825-835"),
    ("N045", "v3_proxy", "proxy descriptor writer identity is zero", "825-835"),
    ("N046", "v3_proxy", "proxy descriptor allocation identity is zero", "825-835"),
    ("N047", "v3_proxy", "proxy descriptor endpoint is none", "825-835"),
    ("N048", "v3_gpu", "GPU producer kind disagrees", "837-846"),
    ("N049", "v3_gpu", "GPU queue endpoint is host pinned", "837-846"),
    ("N050", "v3_gpu", "GPU CQ consumer kind disagrees", "837-846"),
    ("N051", "v3_gpu", "GPU UAR owner is host CPU", "837-846"),
    ("N052", "v2", "host_memory has an extra field", "853-862"),
    ("N053", "v2", "host_memory enabled is false", "863-866"),
    ("N054", "v2", "host-memory device owner is zero", "867-870"),
    ("N055", "v2", "registry field set is incomplete", "872-878"),
    ("N056", "v2", "positive registry field is zero", "879-884"),
    ("N057", "v2", "registry offset reaches 4096", "885-891"),
    ("N058", "v2", "translation path exceeds uint32", "893-898"),
    ("N059", "v2", "translation path is not host pinned", "899-905"),
    ("N060", "v2", "host WQ binding field set is incomplete", "907-913"),
    ("N061", "v2", "positive host WQ binding field is zero", "914-918"),
    ("N062", "v2", "allocation list is empty", "920-925"),
    ("N063", "v2", "allocation field set is incomplete", "931-949"),
    ("N064", "v2", "non-data allocation carries mkey", "931-949"),
    ("N065", "v2", "allocation identity is zero", "950-955"),
    ("N066", "v2", "allocation identities are not ascending", "950-956"),
    ("N067", "v2", "allocation device owner is zero", "958-963"),
    ("N068", "v2", "allocation length is zero", "958-963"),
    ("N069", "v2", "allocation object owner is zero", "958-963"),
    ("N070", "v2", "allocation path identity is zero", "958-963"),
    ("N071", "v2", "data-region mkey is zero", "964-968"),
    ("N072", "v2", "allocation owner kind disagrees", "969-987"),
    ("N073", "v2", "allocation object kind is unknown", "971-987"),
    ("N074", "v2", "allocation endpoint is device memory", "989-994"),
    ("N075", "v2", "QPC allocation is GPU memory", "995-997"),
    ("N076", "v2", "allocation path exceeds uint32", "1004-1008"),
    ("N077", "v2", "allocation path endpoint disagrees", "1009-1014"),
    ("N078", "v2", "page geometry field set is incomplete", "1016-1019"),
    ("N079", "v2", "page size is below 4096", "1020-1024"),
    ("N080", "v2", "page size is not a power of two", "1020-1024"),
    ("N081", "v2", "physical page list is empty", "1025-1031"),
    ("N082", "v2", "physical page is misaligned", "1032-1040"),
    ("N083", "v2", "physical page is duplicated", "1032-1040"),
    ("N084", "v2", "QPC binding allocation is missing", "1043-1059"),
    ("N085", "v2", "SQ binding allocation has wrong kind", "1043-1059"),
    ("N086", "v3_gpu", "SQ allocation endpoint disagrees with submission", "1060-1068"),
    ("N087", "v3_gpu", "CQ allocation endpoint disagrees with submission", "1060-1068"),
    ("N088", "v3_gpu", "doorbell endpoint disagrees with submission", "1060-1068"),
    ("N089", "v3_proxy", "proxy descriptor allocation is missing", "1070-1081"),
    ("N090", "v3_proxy", "proxy has two descriptor allocations", "1070-1081"),
    ("N091", "v3_proxy", "proxy descriptor allocation has wrong kind", "1070-1081"),
    ("N092", "v3_proxy", "proxy descriptor owner disagrees", "1070-1081"),
    ("N093", "v3_proxy", "proxy descriptor endpoint is GPU memory", "1070-1081"),
    ("N094", "v3_host", "non-proxy carries a descriptor allocation", "1082-1085"),
    ("N095", "v2", "work_queue has an extra service field", "1162-1183"),
    ("N096", "v2", "SQ depth is zero", "1184-1186"),
    ("N097", "v2", "CQ depth is zero", "1187-1189"),
    ("N098", "v2", "queue service exceeds int64", "1190-1198"),
    ("N099", "v2", "hardware hash is not lowercase SHA-256", "1955-1974"),
    ("N100", "v2", "hardware hash disagrees with canonical bytes", "1975-1980"),
)


def _configured_root() -> Path:
    raw = os.environ.get("SIMLLM_WAVE5_RUN_ROOT")
    if not raw:
        raise ValueError("SIMLLM_WAVE5_RUN_ROOT must name the external branch run root")
    return Path(raw).resolve()


def _validate_registry(out: Path) -> dict[str, object]:
    try:
        out.resolve().relative_to(_configured_root())
    except ValueError as error:
        raise ValueError("study output must remain under SIMLLM_WAVE5_RUN_ROOT") from error
    bases = {
        "v2": _effective_v2(),
        "v3_host": _effective_v3("host_cpu_driver"),
        "v3_proxy": _effective_v3("cpu_proxy"),
        "v3_gpu": _effective_v3("gpu_initiated"),
    }
    actual_hashes = {name: _digest(value) for name, value in bases.items()}
    if actual_hashes != EFFECTIVE_HASHES:
        raise AssertionError(f"native-emitted fixture hashes drifted: {actual_hashes}")
    v1 = _effective_v1()
    if _digest(v1) != V1_EFFECTIVE_HASH:
        raise AssertionError("v1 effective-hardware hash drifted")
    if _digest(_config(v1)) != V1_CONFIG_HASH:
        raise AssertionError("v1 structural config bytes drifted")
    if _digest(_config(None, bypass=True)) != BYPASS_CONFIG_HASH:
        raise AssertionError("bypass config bytes drifted")
    identifiers = tuple(row[0] for row in MUTATIONS)
    if len(MUTATIONS) != 100 or len(set(identifiers)) != 100:
        raise AssertionError("frozen mutation registry must contain N001 through N100")
    if identifiers != tuple(f"N{index:03d}" for index in range(1, 101)):
        raise AssertionError("frozen mutation identifiers are not contiguous")
    if any(row[1] not in bases for row in MUTATIONS):
        raise AssertionError("mutation references an unknown base fixture")
    sources = (
        REPO_ROOT / "simllm/backends/rnic/src/session_record.cpp",
        REPO_ROOT / "simllm/backends/rnic/include/simllm/rnic/session_record.h",
        REPO_ROOT / "simllm/backends/rnic/tests/submission_test.cpp",
        REPO_ROOT / "simllm/backends/rnic_records.py",
        REPO_ROOT / "tests/test_rnic_records.py",
    )
    missing = [str(path.relative_to(REPO_ROOT)) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen source-audit inputs are missing: {missing}")
    return {
        "artifacts_created": False,
        "bases": {name: EFFECTIVE_HASHES[name] for name in sorted(EFFECTIVE_HASHES)},
        "bypass_config_sha256": BYPASS_CONFIG_HASH,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "mutation_count": len(MUTATIONS),
        "mutation_ids": [row[0] for row in MUTATIONS],
        "simllm_base_commit": SIMLLM_BASE_COMMIT,
        "v1_config_sha256": V1_CONFIG_HASH,
        "v1_effective_sha256": V1_EFFECTIVE_HASH,
    }


def _mapping_at(value: dict[str, Any], *path: str | int) -> Any:
    current: Any = value
    for part in path:
        current = current[part]
    return current


def _set_at(value: dict[str, Any], path: tuple[str | int, ...], replacement: Any) -> None:
    parent = _mapping_at(value, *path[:-1])
    parent[path[-1]] = replacement


_SIMPLE_SETS: dict[str, tuple[tuple[str | int, ...], Any]] = {
    "N001": (("schema",), "simllm-rnic-effective-hardware-v4"),
    "N005": (("unexpected",), True),
    "N006": (("network", "unexpected"), True),
    "N007": (("qpc", "enabled"), 1),
    "N009": (("qpc", "enabled"), False),
    "N010": (("dma", "unexpected"), True),
    "N011": (("dma", "fabric_scope"), "borrowed"),
    "N013": (("dma", "fabric", "generation"), 6),
    "N014": (("dma", "fabric", "lane_count"), 3),
    "N015": (("dma", "fabric", "max_payload_size_bytes"), 64),
    "N016": (("dma", "fabric", "read_completion_boundary_bytes"), 32),
    "N017": (("dma", "fabric", "completion_overhead_bytes"), 0),
    "N018": (("dma", "fabric", "data_credit_unit_bytes"), 3),
    "N019": (("dma", "fabric", "host_to_device_credits", "posted_header_credits"), 0),
    "N020": (
        ("dma", "fabric", "host_to_device_credits", "completion_header_credits"),
        1 << 32,
    ),
    "N021": (("dma", "fabric", "host_to_device_credits", "posted_data_credits"), 1),
    "N022": (("dma", "fabric", "host_store_latency_ps"), [0, 0]),
    "N023": (("dma", "fabric", "paths"), []),
    "N024": (("dma", "fabric", "paths", 1, "path_id"), 1),
    "N025": (("dma", "fabric", "paths", 0, "path_id"), 0),
    "N026": (("dma", "fabric", "paths", 2, "path_id"), 1 << 32),
    "N028": (("dma", "fabric", "paths", 1, "endpoint"), "system_memory"),
    "N029": (("dma", "fabric", "paths", 1, "base_latency_ps"), 1 << 63),
    "N030": (("dma", "fabric", "paths", 2, "enabled"), False),
    "N032": (("dma", "work_queue", "pcie_wqe_bytes"), 0),
    "N033": (("dma", "work_queue", "pcie_cq_first_byte_offset"), 4096),
    "N034": (("dma", "work_queue", "pcie_cq_memory_path_id"), 1 << 32),
    "N035": (("dma", "work_queue", "pcie_cq_memory_path_id"), 999),
    "N036": (("dma", "work_queue", "pcie_cq_memory_path_id"), 1),
    "N037": (("submission", "cq_consumer_ids"), [8101]),
    "N038": (("submission", "producer_id"), 0),
    "N039": (("submission", "cq_consumer_id"), 1 << 32),
    "N040": (("submission", "rnic_requester_id"), 0),
    "N041": (("submission", "producer_shape"), "fpga_proxy"),
    "N042": (("submission", "producer_kind"), "gpu"),
    "N043": (("submission", "descriptor_writer_id"), 1),
    "N044": (("submission", "descriptor_writer_kind"), "host_cpu_driver"),
    "N045": (("submission", "descriptor_writer_id"), 0),
    "N046": (("submission", "descriptor_queue_allocation_id"), 0),
    "N047": (("submission", "descriptor_queue_endpoint"), "none"),
    "N048": (("submission", "producer_kind"), "cpu_proxy"),
    "N049": (("submission", "queue_endpoint"), "host_pinned_memory"),
    "N050": (("submission", "cq_consumer_kind"), "cpu_proxy"),
    "N051": (("submission", "uar_mapping_owner"), "host_cpu"),
    "N052": (("host_memory", "unexpected"), True),
    "N053": (("host_memory", "enabled"), False),
    "N054": (("host_memory", "device_owner_id"), 0),
    "N056": (("host_memory", "registry", "mpt_entry_bytes"), 0),
    "N057": (("host_memory", "registry", "mpt_first_byte_offset"), 4096),
    "N058": (("host_memory", "registry", "translation_path_id"), 1 << 32),
    "N059": (("host_memory", "registry", "translation_path_id"), 3),
    "N061": (("host_memory", "work_queue", "qpc_context_bytes"), 0),
    "N062": (("host_memory", "allocations"), []),
    "N064": (("host_memory", "allocations", 0, "mkey"), 1),
    "N065": (("host_memory", "allocations", 0, "allocation_id"), 0),
    "N067": (("host_memory", "allocations", 0, "device_owner_id"), 0),
    "N068": (("host_memory", "allocations", 0, "length_bytes"), 0),
    "N069": (("host_memory", "allocations", 0, "owner_id"), 0),
    "N070": (("host_memory", "allocations", 0, "path_id"), 0),
    "N071": (("host_memory", "allocations", 5, "mkey"), 0),
    "N072": (("host_memory", "allocations", 0, "owner_kind"), "memory_region"),
    "N073": (("host_memory", "allocations", 0, "object_kind"), "unknown_object"),
    "N074": (("host_memory", "allocations", 5, "endpoint"), "device_memory"),
    "N076": (("host_memory", "allocations", 0, "path_id"), 1 << 32),
    "N077": (("host_memory", "allocations", 0, "path_id"), 3),
    "N079": (("host_memory", "allocations", 0, "pages", "page_size_bytes"), 2048),
    "N080": (("host_memory", "allocations", 0, "pages", "page_size_bytes"), 6144),
    "N081": (("host_memory", "allocations", 0, "pages", "physical_page_addresses"), []),
    "N084": (("host_memory", "work_queue", "qpc_icm_allocation_id"), 999),
    "N085": (("host_memory", "work_queue", "sq_ring_allocation_id"), 24),
    "N092": (("host_memory", "allocations", 6, "owner_id"), 7203),
    "N095": (("work_queue", "cqe_write_service_ps"), 0),
    "N096": (("work_queue", "sq_depth"), 0),
    "N097": (("work_queue", "cq_depth"), 0),
    "N098": (("work_queue", "scheduler_service_ps"), 1 << 63),
}


_REMOVALS: dict[str, tuple[str | int, ...]] = {
    "N002": ("host_memory",),
    "N004": ("submission",),
    "N012": ("dma", "fabric", "analytical_seed"),
    "N027": ("dma", "fabric", "paths", 1, "endpoint"),
    "N031": ("dma", "work_queue", "pcie_wqe_bytes"),
    "N055": ("host_memory", "registry", "mpt_entry_bytes"),
    "N060": ("host_memory", "work_queue", "qpc_context_bytes"),
    "N063": ("host_memory", "allocations", 0, "owner_id"),
    "N078": (
        "host_memory",
        "allocations",
        0,
        "pages",
        "physical_page_addresses",
    ),
}


def _allocation_by_id(value: dict[str, Any], allocation_id: int) -> dict[str, Any]:
    return next(
        allocation
        for allocation in value["host_memory"]["allocations"]
        if allocation["allocation_id"] == allocation_id
    )


def _mutated_effective(
    mutation_id: str,
    base_name: str,
    bases: dict[str, dict[str, object]],
) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = copy.deepcopy(bases[base_name])
    digest_override: str | None = None
    if mutation_id in _SIMPLE_SETS:
        path, replacement = _SIMPLE_SETS[mutation_id]
        _set_at(value, path, copy.deepcopy(replacement))
    elif mutation_id in _REMOVALS:
        path = _REMOVALS[mutation_id]
        parent = _mapping_at(value, *path[:-1])
        parent.pop(path[-1])
    elif mutation_id == "N003":
        value["submission"] = copy.deepcopy(bases["v3_host"]["submission"])
    elif mutation_id == "N008":
        value["dma"] = {"enabled": False}
    elif mutation_id == "N066":
        allocations = value["host_memory"]["allocations"]
        allocations[0], allocations[1] = allocations[1], allocations[0]
    elif mutation_id == "N075":
        allocation = _allocation_by_id(value, 21)
        allocation["endpoint"] = "gpu_memory"
        allocation["path_id"] = 3
    elif mutation_id == "N082":
        pages = _allocation_by_id(value, 21)["pages"]["physical_page_addresses"]
        pages[0] += 1
    elif mutation_id == "N083":
        pages = _allocation_by_id(value, 26)["pages"]["physical_page_addresses"]
        pages[1] = pages[0]
    elif mutation_id in {"N086", "N087", "N088"}:
        allocation_id = {"N086": 22, "N087": 24, "N088": 25}[mutation_id]
        allocation = _allocation_by_id(value, allocation_id)
        allocation["endpoint"] = "host_pinned_memory"
        allocation["path_id"] = 2
    elif mutation_id == "N089":
        value["host_memory"]["allocations"] = [
            allocation
            for allocation in value["host_memory"]["allocations"]
            if allocation["allocation_id"] != 27
        ]
    elif mutation_id == "N090":
        value["host_memory"]["allocations"].append(
            _allocation(
                28,
                "descriptor_queue",
                "submission_producer",
                7202,
                "host_pinned_memory",
                2,
                8,
                1,
                4096,
            )
        )
    elif mutation_id == "N091":
        allocation = _allocation_by_id(value, 27)
        allocation["object_kind"] = "sq_ring"
        allocation["owner_kind"] = "send_queue"
    elif mutation_id == "N093":
        allocation = _allocation_by_id(value, 27)
        allocation["endpoint"] = "gpu_memory"
        allocation["path_id"] = 3
    elif mutation_id == "N094":
        descriptor = copy.deepcopy(_allocation_by_id(bases["v3_proxy"], 27))
        value["host_memory"]["allocations"].append(descriptor)
    elif mutation_id == "N099":
        digest_override = "A" * 64
    elif mutation_id == "N100":
        digest_override = "0" * 64
    else:
        raise AssertionError(f"mutation implementation is missing: {mutation_id}")
    return value, digest_override or _digest(value)


def _native_executable(build_dir: Path) -> Path:
    candidates = (
        build_dir / "simllm_rnic_session_record_test",
        build_dir / "simllm_rnic_session_record_test.exe",
        build_dir / "Release" / "simllm_rnic_session_record_test",
        build_dir / "Release" / "simllm_rnic_session_record_test.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("native session-record probe was not built")


def _build_native(out: Path) -> tuple[Path, str]:
    source = REPO_ROOT / "simllm" / "backends" / "rnic"
    build = out / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSIMLLM_RNIC_BUILD_TESTS=ON",
            "-DSIMLLM_RNIC_BUILD_TOOLS=ON",
            "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "--parallel"],
        check=True,
    )
    ctest = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build),
            "-C",
            "Release",
            "--output-on-failure",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _native_executable(build), ctest.stdout


def _native_observation(
    executable: Path,
    effective_path: Path,
    digest: str,
) -> dict[str, object]:
    completed = subprocess.run(
        [str(executable), "--validate-effective-hardware", str(effective_path), digest],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"native probe infrastructure failed with {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return {
        "accepted": completed.returncode == 0,
        "diagnostic": completed.stderr.strip() or completed.stdout.strip(),
        "returncode": completed.returncode,
    }


def _thaw(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {name: _thaw(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _python_observation(effective: dict[str, Any], digest: str) -> dict[str, object]:
    from simllm.backends.rnic_records import rnic_session_config_from_json

    config = _config(effective)
    config["hardware_config_sha256"] = digest
    try:
        parsed = rnic_session_config_from_json(config)
    except (TypeError, ValueError) as error:
        return {
            "accepted": False,
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    return {
        "accepted": True,
        "diagnostic": "accepted",
        "projection_matches_input": _thaw(parsed.effective_hardware) == effective,
    }


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_study(arguments: argparse.Namespace, plan: dict[str, object]) -> None:
    from simllm.backends import RnicHardwareMode, RnicWqeAuthority
    from simllm.backends.rnic_records import rnic_session_config_from_json

    if arguments.out.exists():
        raise FileExistsError("--out must not exist; choose a fresh external directory")
    arguments.out.mkdir(parents=True)
    input_dir = arguments.out / "inputs"
    input_dir.mkdir()
    executable, ctest_output = _build_native(arguments.out)
    bases: dict[str, dict[str, object]] = {
        "v2": _effective_v2(),
        "v3_host": _effective_v3("host_cpu_driver"),
        "v3_proxy": _effective_v3("cpu_proxy"),
        "v3_gpu": _effective_v3("gpu_initiated"),
    }

    fatal_failures = []
    accepted_controls = []
    frozen_records = []
    for name, effective in bases.items():
        path = input_dir / f"accepted-{name}.json"
        path.write_bytes(_canonical_bytes(effective))
        digest = EFFECTIVE_HASHES[name]
        native = _native_observation(executable, path, digest)
        python = _python_observation(effective, digest)
        passed = bool(
            native["accepted"]
            and python["accepted"]
            and python.get("projection_matches_input")
        )
        accepted_controls.append(
            {
                "fixture": name,
                "native": native,
                "passed": passed,
                "python": python,
                "sha256": digest,
            }
        )
        if not passed:
            fatal_failures.append(f"valid acceptance control {name}")
        if python["accepted"]:
            parsed = rnic_session_config_from_json(_config(effective))
            try:
                parsed.effective_hardware["schema"] = "mutated"
            except TypeError:
                mapping_frozen = True
            else:
                mapping_frozen = False
            try:
                parsed.effective_hardware["host_memory"]["allocations"].append({})
            except AttributeError:
                array_frozen = True
            else:
                array_frozen = False
            frozen_records.append(
                {
                    "array_frozen": array_frozen,
                    "fixture": name,
                    "mapping_frozen": mapping_frozen,
                }
            )
            if not mapping_frozen or not array_frozen:
                fatal_failures.append(f"recursive freeze control {name}")

    v1 = _effective_v1()
    v1_record = rnic_session_config_from_json(_config(v1))
    bypass_record = rnic_session_config_from_json(_config(None, bypass=True))
    off_path = {
        "bypass": {
            "authority": bypass_record.authority.value,
            "config_sha256": _digest(_config(None, bypass=True)),
            "effective_hardware_is_null": bypass_record.effective_hardware is None,
            "hardware_hash_is_null": bypass_record.hardware_config_sha256 is None,
            "mode": bypass_record.hardware_mode.value,
        },
        "v1": {
            "authority": v1_record.authority.value,
            "config_sha256": _digest(_config(v1)),
            "effective_hardware_matches": _thaw(v1_record.effective_hardware) == v1,
            "effective_sha256": _digest(v1),
            "mode": v1_record.hardware_mode.value,
        },
    }
    off_path_passed = (
        off_path["v1"]["config_sha256"] == V1_CONFIG_HASH
        and off_path["v1"]["effective_sha256"] == V1_EFFECTIVE_HASH
        and off_path["v1"]["effective_hardware_matches"]
        and v1_record.hardware_mode is RnicHardwareMode.STRUCTURAL
        and v1_record.authority is RnicWqeAuthority.NATIVE
        and off_path["bypass"]["config_sha256"] == BYPASS_CONFIG_HASH
        and off_path["bypass"]["effective_hardware_is_null"]
        and off_path["bypass"]["hardware_hash_is_null"]
        and bypass_record.hardware_mode is RnicHardwareMode.BYPASS
        and bypass_record.authority is RnicWqeAuthority.ATLAHS_LEDGER
    )
    if not off_path_passed:
        fatal_failures.append("v1 structural or bypass off-path identity")

    raw_observations = []
    for mutation_id, base_name, description, native_lines in MUTATIONS:
        effective, digest = _mutated_effective(mutation_id, base_name, bases)
        path = input_dir / f"{mutation_id}.json"
        path.write_bytes(_canonical_bytes(effective))
        native = _native_observation(executable, path, digest)
        python = _python_observation(effective, digest)
        raw_observations.append(
            {
                "base": base_name,
                "description": description,
                "effective_sha256": _digest(effective),
                "id": mutation_id,
                "native": native,
                "native_source_lines": native_lines,
                "python": python,
                "supplied_hash": digest,
            }
        )

    scored = []
    for observation in raw_observations:
        native_accept = bool(observation["native"]["accepted"])
        python_accept = bool(observation["python"]["accepted"])
        checks = {
            "native_rejected": not native_accept,
            "python_minus_native_exact_zero": int(python_accept) - int(native_accept) == 0,
            "python_rejected": not python_accept,
        }
        scored.append(
            {
                "base": observation["base"],
                "checks": checks,
                "genuine_risk": True,
                "id": observation["id"],
                "native_accept": int(native_accept),
                "passed": all(checks.values()),
                "python_accept": int(python_accept),
                "python_minus_native": int(python_accept) - int(native_accept),
            }
        )
    failed_scored = [row for row in scored if not row["passed"]]
    summary = {
        "schema": "simllm-rnic-records-v3-study-v1",
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "implementation_commit": _git_revision(),
            "simllm_base_commit": SIMLLM_BASE_COMMIT,
        },
        "entailment_analysis": {
            "conclusion": "not entailed",
            "detail": (
                "native and Python outcomes are recorded before scoring; valid-base "
                "acceptance and off-path controls do not constrain any mutated result"
            ),
        },
        "fatal_unscored": {
            "accepted_controls": accepted_controls,
            "ctest_tail": ctest_output.strip().splitlines()[-8:],
            "failures": fatal_failures,
            "off_path": off_path,
            "off_path_passed": off_path_passed,
            "recursive_freeze": frozen_records,
            "passed": not fatal_failures,
        },
        "fixture_hashes": plan["bases"],
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "raw_observations": raw_observations,
        "scored_rejection_family": {
            "genuine_risk_passed": sum(
                bool(row["passed"] and row["genuine_risk"]) for row in scored
            ),
            "genuine_risk_total": sum(bool(row["genuine_risk"]) for row in scored),
            "instances": scored,
            "name": "BACK-28 native/Python effective-hardware rejection parity",
            "passed": sum(bool(row["passed"]) for row in scored),
            "total": len(scored),
        },
    }
    summary_path = arguments.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary={summary_path}")
    print(
        f"fatal={'PASS' if not fatal_failures else 'FAIL'} "
        f"scored={summary['scored_rejection_family']['passed']}/{len(scored)}"
    )
    if fatal_failures:
        raise AssertionError(f"fatal controls failed: {fatal_failures}")
    if failed_scored:
        raise AssertionError(f"scored rejection failures: {failed_scored}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    plan = _validate_registry(arguments.out)
    if arguments.check_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    _run_study(arguments, plan)


if __name__ == "__main__":
    main()
