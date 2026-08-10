from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from simllm.backends import (
    BypassArtifactPaths,
    BypassArtifacts,
    RnicHardwareMode,
    RnicWqeAuthority,
    assert_bypass_artifact_identity,
    canonical_bypass_parameters,
    compare_bypass_artifacts,
    read_bypass_artifacts,
    rnic_bookkeeping_projection_from_json,
    rnic_session_config_from_json,
    rnic_session_result_from_json,
)


def _effective_hardware() -> dict[str, object]:
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


def _disabled_penalties() -> dict[str, object]:
    return {
        name: {"kind": "disabled"}
        for name in ("acs", "ddio_miss", "gpu_direct", "iommu", "numa", "switch_path")
    }


def _dma_effective_hardware() -> dict[str, object]:
    credits = {
        "completion_data_credits": 4096,
        "completion_header_credits": 64,
        "nonposted_header_credits": 64,
        "posted_data_credits": 4096,
        "posted_header_credits": 64,
    }
    return {
        "dma": {
            "enabled": True,
            "fabric": {
                "analytical_seed": 0,
                "completion_buffer_bytes": 65536,
                "completion_buffer_release_latency_ps": 0,
                "completion_overhead_bytes": 20,
                "credit_return_latency_ps": 0,
                "data_credit_unit_bytes": 16,
                "device_to_host_credits": dict(credits),
                "generation": 5,
                "host_store_latency_ps": [0],
                "host_to_device_credits": dict(credits),
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
                        "endpoint": "mmio_bar",
                        "path_id": 1,
                    },
                    {
                        "analytical_penalties": _disabled_penalties(),
                        "base_latency_ps": 0,
                        "enabled": True,
                        "endpoint": "host_pinned_memory",
                        "path_id": 2,
                    },
                ],
                "posted_write_overhead_bytes": 24,
                "posted_write_visibility_latency_ps": [0],
                "read_completion_boundary_bytes": 64,
                "read_completion_latency_ps": [0],
                "read_request_overhead_bytes": 24,
            },
            "fabric_scope": "owned",
            "work_queue": {
                "pcie_completion_ordering_domain": 86,
                "pcie_cq_first_byte_offset": 0,
                "pcie_cq_memory_path_id": 2,
                "pcie_cqe_bytes": 64,
                "pcie_doorbell_record_bytes": 4,
                "pcie_doorbell_record_first_byte_offset": 0,
                "pcie_doorbell_record_path_id": 2,
                "pcie_sq_first_byte_offset": 0,
                "pcie_sq_memory_path_id": 2,
                "pcie_submission_ordering_domain": 83,
                "pcie_uar_doorbell_bytes": 8,
                "pcie_uar_first_byte_offset": 0,
                "pcie_uar_path_id": 1,
                "pcie_wqe_bytes": 64,
            },
        },
        "network": {"enabled": False},
        "qpc": {"enabled": True},
        "schema": "simllm-rnic-effective-hardware-v1",
        "work_queue": {
            "cq_depth": 64,
            "qpc_lookup_service_ps": 5,
            "scheduler_service_ps": 7,
            "sq_depth": 64,
        },
    }


def _digest(value: dict[str, object]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _config(*, bypass: bool = False) -> dict[str, object]:
    effective = None if bypass else _effective_hardware()
    return {
        "authority": "AtlahsWqeLedger" if bypass else "SimllmNativeRnicSession",
        "effective_hardware": effective,
        "hardware_config_sha256": None if bypass else _digest(effective),
        "hardware_mode": "bypass" if bypass else "structural",
        "schema": "simllm-rnic-session-config-v1",
        "session_id": "session-bypass" if bypass else "session-native",
        "transport_policy": "rnic-nn-fluid" if bypass else "rnic-nn",
    }


def _dma_config() -> dict[str, object]:
    value = _config()
    effective = _dma_effective_hardware()
    value["effective_hardware"] = effective
    value["hardware_config_sha256"] = _digest(effective)
    return value


def _timeline() -> dict[str, int | None]:
    return {
        "admitted_at_ps": 60,
        "cqe_visible_at_ps": 73,
        "doorbell_seen_at_ps": 37,
        "doorbelled_at_ps": 0,
        "first_packet_at_ps": None,
        "last_packet_at_ps": None,
        "network_accepted_at_ps": 60,
        "network_outcome_at_ps": 60,
        "polled_at_ps": 73,
        "posted_at_ps": 0,
        "qpc_ready_at_ps": 53,
        "sq_reclaimed_at_ps": 73,
        "transport_retired_at_ps": 60,
        "wqe_fetch_begin_at_ps": 37,
        "wqe_fetch_end_at_ps": 48,
    }


def _wqe(*, bypass: bool = False) -> dict[str, object]:
    return {
        "completion_status": "success",
        "cq_consume_sequence": 1,
        "cq_id": 43,
        "cq_producer_index": 0,
        "cqe_sequence": 1,
        "destination": 4,
        "flow_id": 0,
        "flow_tag": 7,
        "key": {
            "endpoint": 3,
            "post_sequence": 1,
            "session_id": "session-bypass" if bypass else "session-native",
            "wq_id": 41,
            "wq_kind": "send",
        },
        "opcode": "send",
        "payload_bytes": 4096,
        "qpn": None if bypass else 17,
        "signaled": True,
        "source": 3,
        "state": "completed",
        "timeline": _timeline(),
        "transport_kind": "none",
        "transport_object_id": 0,
        "wqe_id": 1,
        "wr_id": None if bypass else 100,
    }


def _completion(*, bypass: bool = False) -> dict[str, object]:
    return {
        "completion_time_ps": 60,
        "cq_consume_sequence": 1,
        "cq_id": 43,
        "cq_post_sequence": 1,
        "destination": 4,
        "fct_ps": 60,
        "flow_id": 0,
        "payload_bytes": 4096,
        "profile": "rnic-nn-fluid" if bypass else "rnic-nn",
        "rq_id": 47 if bypass else None,
        "source": 3,
        "sq_dispatch_sequence": 1,
        "sq_id": 41,
        "sq_post_sequence": 1,
        "start_time_ps": 0,
        "tag": 7,
        "transport_kind": "none",
        "transport_object_id": 0,
        "wqe_id": 1,
    }


def _result(*, bypass: bool = False) -> dict[str, object]:
    config = _config(bypass=bypass)
    return {
        "authority": config["authority"],
        "authority_counters": {
            "legacy_ledger_constructed": int(bypass),
            "legacy_mutations": 2 if bypass else 0,
            "native_posts": 0 if bypass else 1,
            "native_session_constructed": int(not bypass),
        },
        "completion_rows": [_completion(bypass=bypass)],
        "hardware_config_sha256": config["hardware_config_sha256"],
        "hardware_mode": config["hardware_mode"],
        "quiescent": True,
        "schema": "simllm-rnic-session-result-v1",
        "session_id": config["session_id"],
        "transport_policy": config["transport_policy"],
        "wqes": [_wqe(bypass=bypass)],
    }


def _bookkeeping(*, bypass: bool = False) -> dict[str, object]:
    config = _config(bypass=bypass)
    return {
        "authority": config["authority"],
        "hardware_config_sha256": config["hardware_config_sha256"],
        "hardware_mode": config["hardware_mode"],
        "schema": "simllm-rnic-bookkeeping-v1",
        "session_id": config["session_id"],
        "wqes": [_wqe(bypass=bypass)],
    }


def _artifacts() -> BypassArtifacts:
    return BypassArtifacts(
        goal_text=b"goal text\n",
        goal_binary=b"\x00goal",
        topology=b"topology bytes\n",
        profile="rnic-nn-fluid",
        seed=7,
        baseline_parameters=canonical_bypass_parameters(
            {"linkspeed_bps": 400_000_000_000}
        ),
        completion_csv=b"profile,flow_id\nrnic-nn-fluid,0\n",
        canonical_completion=b'[{"flow_id":0,"jct_ps":53}]\n',
        step_results=b"[[0,0,53]]\n",
        replay_summary=b'{"tpot_ps":53,"ttft_ps":53}\n',
    )


def test_config_readers_recompute_hash_and_enforce_mode_exclusivity():
    structural = rnic_session_config_from_json(_config())
    bypass = rnic_session_config_from_json(_config(bypass=True))

    assert structural.hardware_mode is RnicHardwareMode.STRUCTURAL
    assert structural.authority is RnicWqeAuthority.NATIVE
    assert structural.hardware_config_sha256 == _digest(_effective_hardware())
    assert bypass.hardware_mode is RnicHardwareMode.BYPASS
    assert bypass.authority is RnicWqeAuthority.ATLAHS_LEDGER
    assert bypass.hardware_config_sha256 is None
    assert bypass.effective_hardware is None

    wrong_hash = _config()
    wrong_hash["hardware_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        rnic_session_config_from_json(wrong_hash)

    both = _config()
    both["authority"] = "AtlahsWqeLedger"
    with pytest.raises(ValueError, match="structural mode requires"):
        rnic_session_config_from_json(both)

    with pytest.raises(TypeError):
        structural.effective_hardware["work_queue"]["sq_depth"] = 999


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(extra=True), "unknown fields.*extra"),
        (lambda value: value.update(schema="simllm-rnic-session-config-v2"), "schema"),
        (
            lambda value: value["effective_hardware"]["dma"].update(fabric={}),
            "unknown fields.*fabric",
        ),
        (
            lambda value: value["effective_hardware"]["work_queue"].update(
                sq_depth=True
            ),
            "sq_depth",
        ),
    ],
)
def test_config_reader_rejects_schema_shape_and_types(mutation, match):
    value = _config()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        rnic_session_config_from_json(value)


def test_dma_config_reader_round_trips_constructed_device_domain():
    parsed = rnic_session_config_from_json(_dma_config())

    assert parsed.effective_hardware["dma"]["enabled"] is True
    assert parsed.effective_hardware["dma"]["fabric"]["lane_count"] == 16
    with pytest.raises(TypeError):
        parsed.effective_hardware["dma"]["fabric"]["lane_count"] = 8


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda hardware: hardware["dma"]["fabric"].update(lane_count=3),
            "lane_count",
        ),
        (
            lambda hardware: hardware["dma"]["fabric"]["paths"][1].update(
                path_id=1
            ),
            "unique ascending path IDs",
        ),
        (
            lambda hardware: hardware["dma"]["work_queue"].update(
                pcie_cq_memory_path_id=1
            ),
            "incompatible path",
        ),
        (
            lambda hardware: hardware["dma"]["fabric"]["paths"][0][
                "analytical_penalties"
            ].update(
                acs={
                    "incidence_probability_ppm": 1_000_001,
                    "kind": "fixed",
                    "mean_ps": 1,
                }
            ),
            "one million ppm",
        ),
    ],
)
def test_dma_config_reader_rejects_impossible_constructed_shapes(mutation, match):
    value = _dma_config()
    hardware = copy.deepcopy(value["effective_hardware"])
    mutation(hardware)
    value["effective_hardware"] = hardware
    value["hardware_config_sha256"] = _digest(hardware)
    with pytest.raises(ValueError, match=match):
        rnic_session_config_from_json(value)


def test_result_and_bookkeeping_readers_preserve_one_authority_projection():
    structural = rnic_session_result_from_json(_result())
    bypass = rnic_session_result_from_json(_result(bypass=True))
    structural_bookkeeping = rnic_bookkeeping_projection_from_json(_bookkeeping())
    bypass_bookkeeping = rnic_bookkeeping_projection_from_json(
        _bookkeeping(bypass=True)
    )

    assert structural.authority_counters == replace(
        structural.authority_counters,
        native_session_constructed=1,
        legacy_ledger_constructed=0,
        native_posts=1,
        legacy_mutations=0,
    )
    assert bypass.authority_counters.legacy_mutations == 2
    assert structural.wqes == structural_bookkeeping.wqes
    assert bypass.wqes == bypass_bookkeeping.wqes
    assert structural.completion_rows[0].rq_id is None
    assert bypass.completion_rows[0].rq_id == 47
    assert structural.wqes[0].flow_id == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["authority_counters"].update(
                legacy_ledger_constructed=1
            ),
            "mode-exclusivity",
        ),
        (
            lambda value: value["completion_rows"][0].update(rq_id=99),
            "structural send",
        ),
        (
            lambda value: value["completion_rows"][0].update(
                completion_time_ps=61
            ),
            "FCT does not match",
        ),
        (
            lambda value: value["wqes"][0]["timeline"].update(
                network_outcome_at_ps=61, transport_retired_at_ps=61
            ),
            "completion row disagrees",
        ),
        (
            lambda value: value["wqes"][0]["key"].update(endpoint=9),
            "stable key disagrees",
        ),
        (
            lambda value: value["wqes"][0]["timeline"].update(
                qpc_ready_at_ps=47
            ),
            "timestamps must be monotonic",
        ),
        (
            lambda value: value["wqes"][0].update(state="posted"),
            "nonterminal WQE state",
        ),
        (
            lambda value: value["wqes"][0].update(signaled=False),
            "signaling and status must match",
        ),
    ],
)
def test_result_reader_rejects_authority_and_projection_drift(mutation, match):
    value = _result()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        rnic_session_result_from_json(value)


def test_bypass_result_reconciles_exact_ledger_mutation_count():
    value = _result(bypass=True)
    value["authority_counters"]["legacy_mutations"] = 0
    with pytest.raises(ValueError, match="mode-exclusivity"):
        rnic_session_result_from_json(value)


def test_unsignaled_error_completion_retains_required_cqe():
    value = _result()
    value["wqes"][0].update(
        completion_status="transport_error",
        signaled=False,
    )

    parsed = rnic_session_result_from_json(value)
    assert parsed.wqes[0].signaled is False
    assert parsed.wqes[0].completion_status == "transport_error"
    assert parsed.wqes[0].cqe_sequence == 1


def test_cqe_sequence_identity_is_scoped_by_cq():
    value = _result()
    second_wqe = copy.deepcopy(value["wqes"][0])
    second_wqe.update(cq_id=44, flow_id=1, wqe_id=2)
    second_wqe["key"].update(endpoint=5, wq_id=42)
    second_wqe.update(source=5)
    second_completion = copy.deepcopy(value["completion_rows"][0])
    second_completion.update(
        cq_id=44,
        flow_id=1,
        source=5,
        sq_id=42,
        wqe_id=2,
    )
    value["wqes"].append(second_wqe)
    value["completion_rows"].append(second_completion)
    value["authority_counters"]["native_posts"] = 2

    parsed = rnic_session_result_from_json(value)
    assert [(wqe.cq_id, wqe.cqe_sequence) for wqe in parsed.wqes] == [
        (43, 1),
        (44, 1),
    ]

    value["wqes"][1]["cq_id"] = 43
    value["completion_rows"][1]["cq_id"] = 43
    with pytest.raises(ValueError, match="duplicate CQE sequence"):
        rnic_session_result_from_json(value)


def test_bypass_checker_compares_only_frozen_inputs_and_artifacts():
    reference = _artifacts()
    assert assert_bypass_artifact_identity(reference, reference).equivalent

    for name in (
        "completion_csv",
        "canonical_completion",
        "step_results",
        "replay_summary",
    ):
        candidate = replace(reference, **{name: getattr(reference, name) + b"!"})
        comparison = compare_bypass_artifacts(reference, candidate)
        assert comparison.changed_artifacts == (name,)
        assert comparison.changed_inputs == ()
        with pytest.raises(ValueError, match=name):
            assert_bypass_artifact_identity(reference, candidate)

    input_mutations = {
        "goal_text": reference.goal_text + b"!",
        "goal_binary": reference.goal_binary + b"!",
        "topology": reference.topology + b"!",
        "profile": "rnic-cn",
        "seed": reference.seed + 1,
        "baseline_parameters": canonical_bypass_parameters(
            {"linkspeed_bps": 200_000_000_000}
        ),
    }
    for name, changed in input_mutations.items():
        comparison = compare_bypass_artifacts(
            reference, replace(reference, **{name: changed})
        )
        assert comparison.changed_inputs == (name,)
        assert comparison.changed_artifacts == ()


def test_bypass_checker_loads_explicit_files_and_requires_bytes(tmp_path):
    reference = _artifacts()
    paths = {}
    for name in (
        "goal_text",
        "goal_binary",
        "topology",
        "completion_csv",
        "canonical_completion",
        "step_results",
        "replay_summary",
    ):
        path = tmp_path / name
        path.write_bytes(getattr(reference, name))
        paths[name] = path

    paths.update(
        profile=reference.profile,
        seed=reference.seed,
        baseline_parameters=reference.baseline_parameters,
    )

    loaded = read_bypass_artifacts(BypassArtifactPaths(**paths))
    assert loaded == reference

    alternate_root = tmp_path / "different-command-paths"
    alternate_root.mkdir()
    alternate_paths = {}
    for name in (
        "goal_text",
        "goal_binary",
        "topology",
        "completion_csv",
        "canonical_completion",
        "step_results",
        "replay_summary",
    ):
        path = alternate_root / f"renamed-{name}"
        path.write_bytes(getattr(reference, name))
        alternate_paths[name] = path
    alternate_paths.update(
        profile=reference.profile,
        seed=reference.seed,
        baseline_parameters=reference.baseline_parameters,
    )
    alternate = read_bypass_artifacts(BypassArtifactPaths(**alternate_paths))
    assert compare_bypass_artifacts(reference, alternate).equivalent

    invalid = replace(reference, completion_csv="not bytes")
    with pytest.raises(TypeError, match="completion_csv"):
        compare_bypass_artifacts(reference, invalid)


def test_bypass_parameters_are_semantic_and_path_free():
    first = canonical_bypass_parameters(
        {"linkspeed_bps": 400_000_000_000, "cn_margin_ppm": 900_000}
    )
    reordered = canonical_bypass_parameters(
        {"cn_margin_ppm": 900_000, "linkspeed_bps": 400_000_000_000}
    )
    assert first == reordered

    for diagnostic in (
        "argv",
        "binary",
        "command",
        "completion_csv",
        "goal_bin",
        "goal_path",
        "profile",
        "random_seed",
        "rnic_profile",
        "seed",
        "topo",
        "topology_file",
    ):
        with pytest.raises(ValueError, match="diagnostic identity"):
            canonical_bypass_parameters({diagnostic: "somewhere"})
    with pytest.raises(TypeError, match="nonnegative int"):
        canonical_bypass_parameters({"fabric": "/tmp/fabric.json"})

    reference = replace(_artifacts(), baseline_parameters=first)
    candidate = replace(_artifacts(), baseline_parameters=reordered)
    assert compare_bypass_artifacts(reference, candidate).equivalent

    changed = replace(
        candidate,
        baseline_parameters=canonical_bypass_parameters(
            {"cn_margin_ppm": 800_000, "linkspeed_bps": 400_000_000_000}
        ),
    )
    assert compare_bypass_artifacts(reference, changed).changed_inputs == (
        "baseline_parameters",
    )
