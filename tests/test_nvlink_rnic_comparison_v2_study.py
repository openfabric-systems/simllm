import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v2"
EXPECTATIONS = STUDY / "expectations.json"
LEGACY_EXPECTATIONS = (
    ROOT / "examples" / "nvlink_rnic_comparison_v1" / "expectations.json"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "nvlink_rnic_comparison_v2_run_study",
        STUDY / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load() -> dict[str, object]:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def test_run_authority_names_the_expectations_only_commit_and_digest():
    runner = _load_runner()

    assert runner.EXPECTATIONS_COMMIT == (
        "8e69696ba22a600a9aefab21c9f5d93e3f977a77"
    )
    assert runner.EXPECTATIONS_SHA256 == hashlib.sha256(
        EXPECTATIONS.read_bytes()
    ).hexdigest()
    assert runner.PINNED_HTSIM_COMMIT == _load()["htsim_authority"]["commit"]


def test_physical_degree_release_cells_remain_identical_to_traf69_projection():
    frozen = _load()
    legacy = json.loads(LEGACY_EXPECTATIONS.read_text(encoding="utf-8"))
    current = {
        (row["degree"], row["size_bytes"]): row
        for row in frozen["workload"]["cells"]
        if row["degree"] <= 3
    }
    old = {
        (row["degree"], row["size_bytes"]): row
        for row in legacy["workload"]["cells"]
    }

    assert current.keys() == old.keys()
    for key in old:
        for field in (
            "wave_service_ps",
            "release_interval_ps",
            "release_jitter_ps",
            "release_schedule_sha256",
        ):
            assert current[key][field] == old[key][field]


def test_fluid_oracle_queues_transfers_within_one_source_class():
    runner = _load_runner()
    rows = [
        {
            "numeric_id": 1,
            "flow_id": "first",
            "wave": 0,
            "source": 0,
            "destination": 3,
            "payload_bytes": 256,
            "released_at_ps": 0,
        },
        {
            "numeric_id": 2,
            "flow_id": "second",
            "wave": 1,
            "source": 0,
            "destination": 3,
            "payload_bytes": 256,
            "released_at_ps": 0,
        },
    ]

    completion = runner._fluid_oracle(rows, 800_000_000_000, 1_656_815_375_008)

    assert completion == {"first": 2560, "second": 5120}


def test_fluid_oracle_divides_the_full_receiver_plateau():
    runner = _load_runner()
    rows = [
        {
            "numeric_id": source + 1,
            "flow_id": f"flow-{source}",
            "wave": 0,
            "source": source,
            "destination": 3,
            "payload_bytes": 256,
            "released_at_ps": 0,
        }
        for source in range(3)
    ]
    destination = 1_656_815_375_008

    completion = runner._fluid_oracle(rows, 800_000_000_000, destination)
    expected = (256 * 8 * 1_000_000_000_000 + destination // 3 - 1) // (
        destination // 3
    )

    assert set(completion.values()) == {expected}
    assert expected == 3709


def test_jain_fairness_is_exact_at_equal_goodput_and_bounded_otherwise():
    runner = _load_runner()

    assert runner._jain([1.0]) == 1.0
    assert runner._jain([3.0, 3.0, 3.0]) == 1.0
    assert 0 < runner._jain([1.0, 2.0, 4.0]) < 1


def test_adapter_uses_pinned_primitives_and_not_the_homogeneous_runtime():
    source = (STUDY / "rnic_transport_schedule.cpp").read_text(encoding="utf-8")

    assert "RnicMaxMinAllocator::allocate" in source
    assert "RnicPacketizedSlotCalendar" in source
    assert "RnicFluidManifold" in source
    assert "RnicPacketizedManifoldRuntime" not in source
    assert "one-active-transfer-per-ordered-pair-class" in source


def test_plotter_carries_every_required_disclosure_and_evidence_class():
    source = (STUDY / "plot_study.py").read_text(encoding="utf-8")

    assert "required_figure_disclosure" in source
    assert "SIMULATED FCT" in source
    assert "MEASURED endpoint plateaus" in source
    assert "DECLARED packet" in source
    assert "STRUCTURAL NV4" in source


def test_implementation_text_is_lf_portable_and_has_no_em_dash():
    for name in (
        "CMakeLists.txt",
        "rnic_transport_schedule.cpp",
        "run_study.py",
        "plot_study.py",
        "publish_study.py",
    ):
        payload = (STUDY / name).read_bytes()
        assert b"\r" not in payload
        assert "\N{EM DASH}" not in payload.decode("utf-8")
        assert b"/data3/" not in payload
        assert b"/home/" not in payload
