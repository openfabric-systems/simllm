"""Composition checks for the merged host cost and collective latency floor.

The two seams landed in the same functions in one integration wave, and each
branch proved only its own disabled path. These tests prove the two disabled
paths together, byte for byte, and pin the composition the enabled path
computes so that a later change cannot move it silently.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    DurationEstimate,
    HostInitiationModel,
    ModelDims,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import (
    B200_NCCL_2_27_LOCAL_PROFILE,
    LEGACY_COLLECTIVE_LATENCY_PROFILE,
)

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/composed_step_budget_v1/run_study.py"
MISSION_PATH = REPOSITORY / "examples/end_to_end_replay_v1/run_study.py"

FABRIC_SERVICE_PS = 2_500_000
PROVIDER_SERVICE_PS = 2_000
TURING_GPU = GPU_ENVELOPES["gtx1660-ti-sm75"]

DIMS = ModelDims(
    num_layers=1,
    hidden_size=1_024,
    intermediate_size=2_048,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=256,
    dtype_bytes=2,
)


class FixedProvider(ComputeProvider):
    """One fixed fused estimate, so the host term is the only compute input."""

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=PROVIDER_SERVICE_PS, bound="measured")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _study():
    return _load(STUDY_PATH, "composed_step_budget_v1")


def _mission():
    return _load(MISSION_PATH, "end_to_end_replay_v1_for_composition")


def _record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "decode-0",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=8,
            )
        ],
    )


def _stub_backend(monkeypatch):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(
        step_sink_module,
        "run_htsim_rnic",
        lambda config: SimpleNamespace(
            job_completion_time_ps=lambda: FABRIC_SERVICE_PS,
            flows=(),
            quiescent=True,
        ),
    )


def _sink(workdir: Path, **overrides) -> HtsimStepSink:
    config = {
        "profile": "rnic-nn-fluid",
        "tp_ranks": (0, 1),
        "dims": DIMS,
        "workdir": workdir,
        "provider": FixedProvider(),
    }
    config.update(overrides)
    return HtsimStepSink(HtsimStepSinkConfig(**config))


def _goal_artifacts(workdir: Path):
    return tuple((path.name, path.read_bytes()) for path in sorted(workdir.glob("*.goal")))


def _published(sink: HtsimStepSink):
    return (sink.outcomes, sink.locality_outcomes, sink.collective_timing_outcomes)


# ---------------------------------------------------- the two off paths ---


@pytest.mark.parametrize(
    "collective_selector",
    [None, LEGACY_COLLECTIVE_LATENCY_PROFILE],
)
def test_both_disabled_paths_together_reproduce_the_baseline_byte_for_byte(
    tmp_path,
    monkeypatch,
    collective_selector,
):
    _stub_backend(monkeypatch)
    baseline = _sink(tmp_path / "baseline")
    together = _sink(
        tmp_path / "together",
        host_model=HostInitiationModel.ideal(),
        collective_latency_profile=collective_selector,
    )

    baseline_result = baseline(_record())
    together_result = together(_record())

    assert together_result == baseline_result
    assert _published(together) == _published(baseline)
    assert _goal_artifacts(tmp_path / "together") == _goal_artifacts(tmp_path / "baseline")
    assert together.collective_timing_outcomes == []


def test_one_enabled_seam_moves_the_step_and_the_other_stays_exact(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    baseline = _sink(tmp_path / "baseline")
    floor_only = _sink(
        tmp_path / "floor",
        collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE.profile_id,
    )
    host_only = _sink(
        tmp_path / "host",
        gpu=TURING_GPU,
        host_model=HostInitiationModel.turing_cuda_graph(1),
    )

    baseline_ps = baseline(_record()).step_latency_ps
    floor_ps = floor_only(_record()).step_latency_ps
    host_ps = host_only(_record()).step_latency_ps

    added_floor = 2 * B200_NCCL_2_27_LOCAL_PROFILE.base_latency_ps(2)
    assert floor_ps == baseline_ps + added_floor
    assert host_ps == baseline_ps + (810_000 - PROVIDER_SERVICE_PS)
    assert host_only.collective_timing_outcomes == []
    assert _goal_artifacts(tmp_path / "floor") == _goal_artifacts(tmp_path / "baseline")
    assert _goal_artifacts(tmp_path / "host") == _goal_artifacts(tmp_path / "baseline")


# --------------------------------------------------- the composed step ---


def test_enabled_composition_adds_the_network_outside_the_host_overlap(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    baseline = _sink(tmp_path / "baseline")
    composed = _sink(
        tmp_path / "composed",
        gpu=TURING_GPU,
        host_model=HostInitiationModel.turing_cuda_graph(1),
        collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE.profile_id,
    )

    baseline_ps = baseline(_record()).step_latency_ps
    composed_ps = composed(_record()).step_latency_ps

    timing = composed.collective_timing_outcomes[0]
    locality = composed.locality_outcomes[0]
    base_sum = sum(row.collective_base_latency_ps for row in timing.artifacts)
    fabric_sum = sum(locality.fabric_phase_service_ps)
    quantized_host_ps = 810_000
    launch_demand_ps = HostInitiationModel.turing_cuda_graph(1).launch_floor_ps

    assert locality.compute_service_ps == quantized_host_ps
    assert composed_ps == quantized_host_ps + base_sum + fabric_sum

    additive = quantized_host_ps + base_sum + fabric_sum
    overlapped = max(baseline_ps + base_sum, launch_demand_ps)
    assert additive != overlapped
    assert composed_ps != overlapped


def test_a_larger_launch_demand_moves_the_composed_step_by_its_whole_difference(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    small = _sink(
        tmp_path / "small",
        gpu=TURING_GPU,
        host_model=HostInitiationModel.turing_cuda_graph(1),
        collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE.profile_id,
    )
    large = _sink(
        tmp_path / "large",
        gpu=TURING_GPU,
        host_model=HostInitiationModel.turing_cuda_graph(3),
        collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE.profile_id,
    )

    small_ps = small(_record()).step_latency_ps
    large_ps = large(_record()).step_latency_ps

    assert large_ps - small_ps == 2_428_000 - 810_000


# ----------------------------------------- the mission runner selection ---


def _namespace(**overrides) -> argparse.Namespace:
    values = {
        "host_profile": "ideal",
        "host_launch_count": 0,
        "collective_latency_profile": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_mission_runner_defaults_select_the_accepted_configuration():
    mission = _mission()

    selection = mission._composition_selection(_namespace())

    assert selection["is_default"] is True
    assert selection["host_model"].is_ideal
    assert selection["gpu"].name == "b100"
    assert selection["collective_latency_profile"] is None


def test_mission_runner_calibrated_selection_shows_the_device_hybrid():
    mission = _mission()

    selection = mission._composition_selection(
        _namespace(
            host_profile="turing-eager-host",
            host_launch_count=567,
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE.profile_id,
        )
    )

    assert selection["is_default"] is False
    assert selection["gpu"].name == "gtx1660-ti-sm75"
    assert selection["provider_envelope"] == "b100"
    assert selection["host_model"].launch_floor_ps == 567 * 2_364_255


def test_pinned_provider_prices_against_the_accepted_envelope():
    mission = _mission()
    from simllm.compute import RooflineProvider, step_kernel

    record = _record()
    dims = mission._dims(8)
    kernel = step_kernel(dims, record, num_sampled=1)
    pinned = mission._pinned_envelope_provider("b100", 0.7)
    reference = RooflineProvider(0.7)

    assert pinned.estimate(kernel, TURING_GPU) == reference.estimate(
        kernel, GPU_ENVELOPES["b100"]
    )
    assert pinned.estimate(kernel, TURING_GPU) != reference.estimate(kernel, TURING_GPU)


def test_mission_runner_rejects_an_ill_formed_selection():
    mission = _mission()

    with pytest.raises(SystemExit, match="positive launch count"):
        mission._composition_selection(_namespace(host_profile="turing-cuda-graph"))
    with pytest.raises(SystemExit, match="takes no launch count"):
        mission._composition_selection(_namespace(host_launch_count=440))
    with pytest.raises(SystemExit, match="unknown host profile"):
        mission._composition_selection(_namespace(host_profile="b100-guess"))


def test_child_command_is_unchanged_by_default_and_carries_the_selection(tmp_path):
    mission = _mission()
    args = argparse.Namespace(
        cache_dir=tmp_path / "cache",
        htsim_rnic=tmp_path / "htsim",
        run_dir=tmp_path / "run",
    )

    default = mission.child_command(args, "cell:a-ep8-400g")
    selected = mission.child_command(
        args,
        "cell:a-ep8-400g",
        cell_label="on-graph440-400g",
        host_profile="turing-cuda-graph",
        host_launch_count=440,
        collective_latency_profile="b200-nccl-2.27-local-v1",
    )

    assert "--cell-label" not in default
    assert "--host-profile" not in default
    assert "--collective-latency-profile" not in default
    assert selected[len(default) :] == [
        "--cell-label",
        "on-graph440-400g",
        "--host-profile",
        "turing-cuda-graph",
        "--host-launch-count",
        "440",
        "--collective-latency-profile",
        "b200-nccl-2.27-local-v1",
    ]


def test_accepted_five_cell_run_refuses_a_nondefault_selection(tmp_path):
    mission = _mission()
    args = argparse.Namespace(
        cache_dir=tmp_path / "cache",
        htsim_rnic=tmp_path / "htsim",
        run_dir=tmp_path / "run",
        cell_label=None,
        host_profile="turing-cuda-graph",
        host_launch_count=440,
        collective_latency_profile=None,
    )

    with pytest.raises(SystemExit, match="accepted configuration only"):
        mission.run_study(args)
    assert not (tmp_path / "run").exists()


# ------------------------------------------------------ relation scoring ---


def _decode_cell(latencies, compositions=None):
    steps = []
    for index, latency in enumerate(latencies):
        composition = (
            compositions[index]
            if compositions is not None
            else [("r00", "decode", 1, 8 + index)]
        )
        steps.append(
            {
                "step_index": index,
                "simulated": True,
                "step_latency_ps": latency,
                "compute_service_ps": 356_095_000,
                "scheduled": [
                    {
                        "request_id": request_id,
                        "phase": phase,
                        "num_new_tokens": tokens,
                        "context_length": context,
                    }
                    for request_id, phase, tokens, context in composition
                ],
            }
        )
    return {"steps": steps}


def _composition_record(latencies, compute_service_ps):
    return {
        "steps": [
            {
                "step_index": index,
                "step_latency_ps": latency,
                "compute_service_ps": compute_service_ps,
            }
            for index, latency in enumerate(latencies)
        ]
    }


def test_f1_fails_when_the_step_lands_in_the_overlapped_interval():
    study = _study()
    frozen = study.check_module().load_expectations()
    label = "on-graph440-400g"
    overlapped = frozen["intervals_ps"]["overlapped"][label]
    midpoint = (overlapped[0] + overlapped[1]) // 2
    cells = {label: _decode_cell([midpoint, midpoint + 1_000])}
    compositions = {label: _composition_record([midpoint, midpoint + 1_000], 99_024_000)}

    result = study._relation_f1(cells, compositions, frozen)

    assert result["passed"] is False
    assert result["rows"][0]["inside_overlapped"] == 2


def test_f1_passes_on_an_additive_shaped_cell():
    study = _study()
    frozen = study.check_module().load_expectations()
    label = "on-graph440-400g"
    additive = frozen["intervals_ps"]["additive"][label]
    midpoint = (additive[0] + additive[1]) // 2
    cells = {label: _decode_cell([midpoint, midpoint + 1_000])}
    compositions = {
        label: _composition_record([midpoint, midpoint + 1_000], 356_095_000)
    }

    result = study._relation_f1(cells, compositions, frozen)

    assert result["passed"] is True
    assert result["rows"][0]["inside_additive"] == 2


def test_f2_reports_not_evaluated_below_the_frozen_matched_minimum():
    study = _study()
    frozen = study.check_module().load_expectations()
    cells = {
        "on-graph440-400g": _decode_cell([1_907_743_126]),
        "on-eager567-400g": _decode_cell(
            [2_892_181_126], compositions=[[("r00", "decode", 1, 99)]]
        ),
    }

    result = study._relation_f2(cells, frozen)

    assert result["evaluated"] is False
    assert result["matched"] == 0


def test_f2_fails_when_the_two_profiles_do_not_separate():
    study = _study()
    frozen = study.check_module().load_expectations()
    cells = {
        "on-graph440-400g": _decode_cell([1_650_672_126, 1_650_672_130]),
        "on-eager567-400g": _decode_cell([1_650_672_126, 1_650_672_130]),
    }

    result = study._relation_f2(cells, frozen)

    assert result["evaluated"] is True
    assert result["passed"] is False
    assert [row["difference_ps"] for row in result["differences"]] == [0, 0]


def test_f3_fails_when_the_halving_is_not_compressed():
    study = _study()
    frozen = study.check_module().load_expectations()
    cells = {
        "on-graph440-400g": _decode_cell([1_000_000_000, 1_000_000_000]),
        "on-graph440-200g": _decode_cell([1_200_000_000, 1_100_000_000]),
    }

    result = study._relation_f3(cells, frozen)

    assert result["evaluated"] is True
    assert result["passed"] is False
    assert result["observed_ratio_range"][1] == pytest.approx(1.2)
