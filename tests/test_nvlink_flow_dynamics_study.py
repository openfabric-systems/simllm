import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from simllm.backends.htsim_nvlink import NvlinkPacket, NvlinkPacketDirection

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_flow_dynamics_v1"
PROFILE = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_study = _load("nvlink_flow_dynamics_plot", "plot_study.py")
publish_study = _load("nvlink_flow_dynamics_publish", "publish_study.py")
run_study = _load("nvlink_flow_dynamics_run", "run_study.py")


def test_runner_pins_the_final_preimplementation_authority():
    assert run_study.EXPECTATIONS_COMMIT == "32a49805546bd038af5e49fd68b5d2ed0cea6174"
    assert run_study.EXPECTATIONS_SHA256 == hashlib.sha256(
        (STUDY / "expectations.json").read_bytes()
    ).hexdigest()
    frozen = run_study.load_expectations()
    assert frozen["study"]["status"] == "expectations_only"
    assert "phase-3 identity" in frozen["study"]["preimplementation_amendment"]


def test_static_identity_and_preservation_projection_are_complete():
    frozen = run_study.load_expectations()
    artifacts = run_study._preservation_artifacts(frozen)

    assert len(artifacts) == frozen["preservation_lock"]["expected_total_artifacts"] == 60
    assert run_study._static_identity(PROFILE) == run_study.STATIC_IDENTITY_SHA256
    assert len({artifact["path"] for artifact in artifacts}) == 60


def test_transition_probe_matches_both_exact_identities():
    frozen = run_study.load_expectations()
    transitions = run_study._run_transitions(frozen, PROFILE)

    assert transitions["convergence"]["observed_open_ps"] == 13886
    assert transitions["convergence"]["residual_ps"] == 0
    assert transitions["convergence"]["verdict"] == "PASS"
    assert transitions["divergence"]["observed_time_to_target_ps"] == 16684
    assert transitions["divergence"]["residual_ps"] == 0
    assert transitions["divergence"]["verdict"] == "PASS"
    assert len(transitions["convergence"]["rate_rows"]) == 178
    assert len(transitions["divergence"]["rate_rows"]) == 322


def test_raw_rate_bins_are_fixed_counts_without_smoothing():
    packets = (
        NvlinkPacket(
            extent_id="flow",
            attempt_id="flow:0",
            sequence=0,
            source=0,
            destination=1,
            direction=NvlinkPacketDirection.REQUEST,
            payload_bytes=256,
            header_bytes=16,
            wire_bytes=272,
            released_at_ps=0,
            delivered_at_ps=10,
        ),
        NvlinkPacket(
            extent_id="flow",
            attempt_id="flow:1",
            sequence=1,
            source=0,
            destination=1,
            direction=NvlinkPacketDirection.REQUEST,
            payload_bytes=128,
            header_bytes=16,
            wire_bytes=144,
            released_at_ps=0,
            delivered_at_ps=150,
        ),
    )

    rows = run_study._rate_rows(packets, ["flow"], bin_ps=100)

    assert [row["payload_bytes"] for row in rows] == [256, 128]
    assert [row["payload_gbps"] for row in rows] == [2560.0, 1280.0]


def test_seeded_release_schedule_is_reproducible_and_seed_sensitive():
    arguments = {
        "degree": 3,
        "size_bytes": 4096,
        "release_interval_ps": 100000,
        "jitter_low_ps": -1000,
        "jitter_high_ps": 1000,
        "samples_per_sender": 12,
    }
    first = run_study._sample_releases(seed=1103, **arguments)
    repeat = run_study._sample_releases(seed=1103, **arguments)
    other = run_study._sample_releases(seed=1907, **arguments)

    assert first == repeat
    assert first != other
    assert len(first) == 36
    assert all(release >= 0 for _, _, release in first)
    assert {source for _, source, _ in first} == {0, 1, 2}


def test_empirical_cdf_mean_and_minmax_band_are_pointwise():
    rows = run_study._cdf_rows({1: [10, 20, 30], 2: [10, 10, 40]})

    assert [row["fct_ps"] for row in rows] == [10, 20, 30, 40]
    assert rows[0] == {
        "fct_ps": 10,
        "cdf_mean": 0.5,
        "cdf_min": 1 / 3,
        "cdf_max": 2 / 3,
    }
    assert rows[-1]["cdf_min"] == rows[-1]["cdf_mean"] == rows[-1]["cdf_max"] == 1


def test_plot_and_publication_contract_names_all_five_figures():
    frozen = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))

    assert frozen["plot_contract"]["formats"] == ["pdf", "png"]
    assert len(publish_study.EXPECTED_FIGURES) == 10
    assert set(publish_study.EXPECTED_FIGURES) == {
        f"{stem}.{suffix}"
        for stem in (
            "nvlink-flow-dynamics",
            "nvlink-fct-cdf",
            "nvlink-incast-degree-1",
            "nvlink-incast-degree-2",
            "nvlink-incast-degree-3",
        )
        for suffix in ("pdf", "png")
    }
    plot_source = (STUDY / "plot_study.py").read_text(encoding="utf-8")
    assert "Raw fixed bins, no smoothing" in plot_source
    assert "pointwise seed min-max" in plot_source
    assert plot_study.DISCLOSURE.startswith("Mixed evidence")


def test_study_modules_have_no_unix_only_module_scope_imports():
    forbidden = {"fcntl", "pwd", "resource", "termios"}
    for path in (
        STUDY / "run_study.py",
        STUDY / "plot_study.py",
        STUDY / "publish_study.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        assert not imports & forbidden
        assert "typing_extensions" not in imports
