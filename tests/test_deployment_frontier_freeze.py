import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.deployment_frontier_v1.frontier import load_expectations, sha256_file
from examples.deployment_frontier_v1.run_study import (
    EXPECTATIONS_COMMIT,
    EXPECTATIONS_SHA256,
)

EXPECTATIONS = ROOT / "examples" / "deployment_frontier_v1" / "expectations.json"
PRE_TRAF70_NVLINK_PROFILE = (
    ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile-pre-traf70.json"
)


def test_expectations_are_bound_to_the_preimplementation_commit():
    assert sha256_file(EXPECTATIONS) == EXPECTATIONS_SHA256
    committed = subprocess.run(
        [
            "git",
            "show",
            f"{EXPECTATIONS_COMMIT}:examples/deployment_frontier_v1/expectations.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(committed).hexdigest() == EXPECTATIONS_SHA256


def test_plot_contract_is_the_new_log_log_frontier_only():
    frozen = load_expectations()
    contract = frozen["plot_contract"]

    assert contract["schema"] == "simllm-deployment-frontier-plot-contract-v2"
    assert contract["prior_flagship_axis_contract_unchanged"] is True
    assert contract["amendment_scope"] == "new batch-sweep frontier figures only"
    assert contract["x"]["quantity"] == "per_request_decode_speed"
    assert contract["y"]["quantity"] == "aggregate_output_throughput_normalized_per_gpu"
    assert contract["x"]["scale"] == contract["y"]["scale"] == "log"
    assert contract["analytical_style"] == "solid line per deployment configuration"
    assert contract["simulation_style"] == "filled dots at every swept batch"
    assert contract["published_y_only_style"] == "dashed horizontal line"


def test_frozen_projection_reconstructs_batch_32_exactly():
    inventory = load_expectations()["model_inventory"]

    assert inventory["flops_per_batch_item"] * 32 == inventory["frozen_flops"]
    assert (
        inventory["static_logical_hbm_bytes"]
        + inventory["dynamic_hbm_bytes_per_batch_item"] * 32
        == inventory["frozen_logical_hbm_bytes"]
    )
    geometry = inventory["network_geometry"]
    assert (
        geometry["moe_layers"]
        * geometry["dispatch_and_combine_phases_per_layer"]
        * geometry["top_k"]
        * geometry["hidden_size"]
        * geometry["element_bytes"]
        == geometry["logical_collective_bytes_per_gpu_per_batch_item"]
    )


def test_all_frozen_sources_and_43_preservation_entries_match():
    frozen = load_expectations()
    checks = [
        (frozen["model_inventory"]["path"], frozen["model_inventory"]["sha256"]),
        (
            frozen["gpu_envelopes"]["source_path"],
            frozen["gpu_envelopes"]["source_sha256"],
        ),
        (
            frozen["gpu_envelopes"]["roofline_source_path"],
            frozen["gpu_envelopes"]["roofline_source_sha256"],
        ),
        (
            frozen["network_inputs"]["fabric"]["source_path"],
            frozen["network_inputs"]["fabric"]["source_sha256"],
        ),
        (
            frozen["network_inputs"]["intra_node"]["profile_path"],
            frozen["network_inputs"]["intra_node"]["profile_sha256"],
        ),
        (
            frozen["network_inputs"]["intra_node"]["implementation_path"],
            frozen["network_inputs"]["intra_node"]["implementation_sha256"],
        ),
        (
            frozen["published_context"]["source_path"],
            frozen["published_context"]["source_sha256"],
        ),
    ]
    for relative, expected in checks:
        path = ROOT / relative
        if relative == "examples/a100_nvlink_packet_v1/candidate-profile.json":
            published = json.loads(path.read_text(encoding="utf-8"))
            assert published["traf70_score_publication"][
                "protected_candidate_before_sha256"
            ] == expected
            path = PRE_TRAF70_NVLINK_PROFILE
        elif relative == "simllm/backends/htsim_nvlink.py":
            frozen_source = subprocess.run(
                ["git", "show", f"{EXPECTATIONS_COMMIT}:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            assert hashlib.sha256(frozen_source).hexdigest() == expected
            continue
        assert sha256_file(path) == expected

    lock = frozen["preservation_lock"]
    inherited_path = ROOT / lock["inherited"]["path"]
    assert sha256_file(inherited_path) == lock["inherited"]["sha256"]
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))[
        "preservation_lock"
    ]["artifacts"]
    artifacts = inherited + lock["additional_artifacts"]
    assert len(inherited) == 33
    assert len(artifacts) == lock["expected_total_artifacts"] == 43
    for artifact in artifacts:
        assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"]


def test_accounting_and_closure_contract_are_literal():
    frozen = load_expectations()

    assert frozen["gpu_envelopes"]["kernel_simulation_enabled"] is False
    assert frozen["gpu_envelopes"]["efficiency"] == 1.0
    assert frozen["accounting_identity"]["residual_ps"].startswith(
        "simulated_step_ps - analytical_step_ps"
    )
    assert "residual_ps == 0" in frozen["accounting_identity"]["pass_rule"]
    assert frozen["study"]["reserved_residual_ids"] == ["TRAF-69", "COMP-77"]
    assert len(frozen["configurations"]) * len(frozen["batch_per_gpu_sweep"]) == 18
