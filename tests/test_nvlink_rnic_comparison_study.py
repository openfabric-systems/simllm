import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v1"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_study = _load("nvlink_rnic_comparison_run", "run_study.py")
plot_study = _load("nvlink_rnic_comparison_plot", "plot_study.py")
publish_study = _load("nvlink_rnic_comparison_publish", "publish_study.py")


def _frozen():
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def test_runner_is_bound_to_the_frozen_authority_and_pinned_runtime():
    assert (
        run_study.EXPECTATIONS_COMMIT
        == "6224d90fea2eed788b8e6ba876787fe7f0e52319"
    )
    assert (
        run_study.EXPECTATIONS_SHA256
        == "4b60365d8251b5fd3c7627dbe38c66ad1fc1c096b21fdfada4fc744320a5bdfa"
    )
    assert (
        run_study.PINNED_HTSIM_COMMIT
        == "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
    )
    assert run_study.RESULT_SCHEMA == "simllm-nvlink-rnic-comparison-result-v1"


def test_every_release_schedule_rebuilds_before_transport_execution():
    frozen = _frozen()
    for cell in frozen["workload"]["cells"]:
        releases = run_study.release_lists(cell, frozen["workload"])
        assert len(releases) == 9
        assert all(
            len(seed_rows)
            == cell["degree"] * frozen["workload"]["samples_per_seed_per_sender"]
            for seed_rows in releases.values()
        )


def test_cdf_and_nearest_rank_dispersion_math_are_exact_and_monotone():
    samples = {1: [10, 20, 30, 40], 2: [10, 20, 40, 50], 3: [20, 30, 40, 60]}
    rows = run_study._cdf_rows(samples)
    summary = run_study._seed_summary(samples)

    assert run_study._quantile(samples[1], 0.50) == 20
    assert run_study._quantile(samples[1], 0.95) == 40
    assert run_study._cdf_valid(rows)
    assert rows[-1] == {
        "fct_ps": 60,
        "cdf_mean": 1.0,
        "cdf_min": 1.0,
        "cdf_max": 1.0,
    }
    assert summary["dispersion_width_ps"] == 10
    assert summary["p50_seed_median_ps"] == 20
    assert summary["dispersion_ratio"] == 0.5


def test_rnic_command_receives_only_the_frozen_physical_mapping(tmp_path):
    frozen = _frozen()
    command = run_study._rnic_command(
        tmp_path / "adapter",
        tmp_path / "schedule.csv",
        tmp_path / "completion.csv",
        tmp_path / "manifest.json",
        1_656_815_375_008,
        frozen["physical_constants"],
    )

    assert command[command.index("--capacity-bps") + 1] == "1656815375008"
    assert command[command.index("--max-wire-bytes") + 1] == "272"
    assert command[command.index("--header-bytes") + 1] == "16"
    assert command[command.index("--propagation-ps") + 1] == "0"
    assert command[command.index("--nodes") + 1] == "4"
    assert not any("ack" in argument.lower() for argument in command)
    assert not any("window" in argument.lower() for argument in command)


def test_adapter_calls_the_pinned_max_min_packet_runtime_without_ack_logic():
    source = (STUDY / "rnic_nn_schedule.cpp").read_text(encoding="utf-8")
    cmake = (STUDY / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "RnicPacketizedManifoldRuntime runtime(" in source
    assert "RnicDataPacketizationConfig" in source
    assert '"ack_events": 0' not in source
    assert '\\"ack_events\\\": 0' in source
    assert "reverse_control_bytes\\\": 0" in source
    assert "rnic_packetized_manifold_runtime.cpp" in cmake
    assert "htsim_logged_minimal.cpp" in cmake
    assert "HTSIM_SOURCE_DIR" in cmake
    assert "HTSIM_SOURCE_COMMIT" in cmake


def test_plot_contract_has_seven_cdf_panels_and_three_dispersion_panels():
    frozen = _frozen()

    assert frozen["plot_contract"]["cdf_panels"] == 7
    assert frozen["plot_contract"]["dispersion_panels"] == 3
    assert plot_study.TRANSPORT_STYLES == {
        "nvlink-credit": "-",
        "rnic-nn": "--",
    }
    assert set(publish_study.EXPECTED_FIGURES) == {
        "nvlink-rnic-fct-cdf.pdf",
        "nvlink-rnic-fct-cdf.png",
        "nvlink-rnic-dispersion.pdf",
        "nvlink-rnic-dispersion.png",
    }


def test_publisher_derives_all_twenty_one_side_by_side_dispersion_cells():
    summaries = []
    for size_index, size_bytes in enumerate(publish_study.SIZE_LABELS, start=1):
        for degree in (1, 2, 3):
            summaries.extend(
                [
                    {
                        "transport": "nvlink-credit",
                        "degree": degree,
                        "size_bytes": size_bytes,
                        "dispersion_ratio": 0.01 * size_index,
                    },
                    {
                        "transport": "rnic-nn",
                        "degree": degree,
                        "size_bytes": size_bytes,
                        "dispersion_ratio": 0.005 * size_index,
                    },
                ]
            )
    rows = publish_study._dispersion_rows({"cell_summaries": summaries})

    assert len(rows) == 21
    assert all(row["tighter_transport"] == "rnic-nn" for row in rows)
    assert all(row["wider_to_tighter_factor"] == 2 for row in rows)


def test_implementation_is_lf_portable_and_has_no_observation_artifacts_yet():
    for filename in (
        "CMakeLists.txt",
        "htsim_logged_minimal.cpp",
        "plot_study.py",
        "publish_study.py",
        "rnic_nn_schedule.cpp",
        "run_study.py",
    ):
        content = (STUDY / filename).read_bytes()
        assert b"\r" not in content
        assert b"\xe2\x80\x94" not in content
        assert b"/data3/" not in content
        assert b"/home/" not in content
        assert b"typing_extensions" not in content
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "RESULTS.md").exists()
    assert not (STUDY / "dispersion.csv").exists()
    assert not (STUDY / "figures").exists()
