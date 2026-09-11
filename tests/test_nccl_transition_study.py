"""Check that the study rejects invalid evidence and keeps tuning unscored."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nccl_transition_v1"
sys.path.insert(0, str(STUDY))
spec = importlib.util.spec_from_file_location("nccl_transition_analysis", STUDY / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
observer = sys.modules["observe_results"]
sys.path.pop(0)


def benchmark_row():
    return {
        "version": 4,
        "nccl_version": 23102,
        "out_of_bounds": {"count": 0},
        "errors": [""],
        "results": [
            {
                "size": 1048576,
                "type": "float",
                "redop": "sum",
                "out_of_place": {"time": 20, "bus_bw": 52.4288, "nwrong": 0},
                "in_place": {"time": 19, "bus_bw": 55.18, "nwrong": 0},
            }
        ],
    }


def test_full_validation_failure_cannot_enter_a_timing_curve():
    data = benchmark_row()
    data["results"][0]["in_place"]["nwrong"] = 1
    with pytest.raises(ValueError, match="F2"):
        analysis.extract_rows(data, {"lane": "timing", "width": 2})


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_nonphysical_time_is_fatal(duration):
    data = benchmark_row()
    data["results"][0]["out_of_place"]["time"] = duration
    with pytest.raises(ValueError, match="F1"):
        analysis.extract_rows(data, {"lane": "timing", "width": 2})


def test_profiler_time_cannot_be_pooled_with_latency():
    common = {
        "architecture": "a100",
        "arm": "auto",
        "width": 2,
        "bytes": 1048576,
        "repeat": 0,
        "sequence": 1,
        "busbw_gbps": 1,
    }
    records = [
        dict(common, lane="timing", time_us=20),
        dict(common, lane="timing", time_us=22),
        dict(common, lane="diagnostic", time_us=500, algo="Ring", proto="Simple"),
    ]
    points, tuning = analysis.summarize(records)
    assert points[0]["median_us"] == 21
    assert points[0]["repetitions"] == 2
    assert "time_us" not in tuning[0]
    assert "busbw_gbps" not in tuning[0]


def test_unexpected_instrumentation_voids_a_timing_row():
    data = benchmark_row()
    data["results"][0]["tuning"] = {"proto": "Simple"}
    with pytest.raises(ValueError, match="F6"):
        analysis.extract_rows(data, {"lane": "timing", "width": 2})


def test_unknown_runtime_cannot_claim_the_frozen_library():
    data = benchmark_row()
    data["nccl_version"] = 22105
    with pytest.raises(ValueError, match="F3"):
        analysis.extract_rows(data, {"lane": "timing", "width": 2})


def test_repeat_spread_is_between_processes_without_tail_trimming():
    assert analysis.quantile([10, 12, 14, 16, 1000], 0.25) == 12
    assert analysis.quantile([10, 12, 14, 16, 1000], 0.75) == 16


def test_dense_plan_covers_original_anomaly_and_its_neighbors():
    config = json.loads((STUDY / "expectations.json").read_text())
    sizes = analysis.grid_values(config, "dense_grid")
    assert len(sizes) == 241
    assert {1048576 - 16384, 1048576, 1048576 + 16384}.issubset(sizes)
    plan = analysis.make_plan(config)
    keys = [(r["lane"], r["arm"], r["width"], r["repeat"]) for r in plan]
    assert len(keys) == len(set(keys))
    for width in (2, 4):
        assert (
            sum(r["lane"] == "timing" and r["arm"] == "auto" and r["width"] == width for r in plan)
            == 5
        )


def callbacks(width=2, sizes=(1024,), placements=(False,)):
    return [
        {
            "kind": "collective",
            "function": "AllReduce",
            "datatype": "ncclFloat32",
            "width": width,
            "rank": rank,
            "sequence": index,
            "count": size // 4,
            "in_place": placement,
            "algo": "RING",
            "proto": "LL",
            "#channels": 8,
            "#warps": 16,
            "kernelVariant": "",
        }
        for index, size in enumerate(sizes)
        for rank in range(width)
        for placement in placements
    ]


@pytest.mark.parametrize("width", [2, 4])
def test_callback_join_requires_every_rank_at_every_payload(width):
    events = callbacks(width, sizes=(1024, 2048))
    with pytest.raises(ValueError, match="missing a participant rank"):
        observer.callback_choices(events[:-1], [width], [1024, 2048], legacy=True)


def test_conflicting_callbacks_are_preserved_instead_of_choosing_one():
    events = callbacks()
    events[-1]["proto"] = "SIMPLE"
    rows, conflicts = observer.callback_choices(events, [2], [1024], legacy=True)
    assert rows == []
    assert {r["proto"] for r in conflicts[0]["choices"]} == {"LL", "SIMPLE"}


def test_buffer_placements_have_independent_selection_records():
    events = callbacks(placements=(False, True))
    for event in events:
        if event["in_place"]:
            event["proto"] = "SIMPLE"
    rows, conflicts = observer.callback_choices(events, [2], [1024])
    assert conflicts == []
    assert {r["in_place"]: r["proto"] for r in rows} == {False: "LL", True: "SIMPLE"}


def test_empty_protocol_is_missing_evidence_even_when_callback_exists():
    events = callbacks()
    events[-1]["proto"] = ""
    with pytest.raises(ValueError, match="choice fields are empty"):
        observer.callback_choices(events, [2], [1024], legacy=True)
