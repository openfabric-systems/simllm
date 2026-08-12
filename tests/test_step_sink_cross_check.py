"""Runtime selection tests for the independent dependency cross-check."""

import hashlib
import re
import time
from pathlib import Path

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimStepSink,
    HtsimStepSinkConfig,
)
from simllm.backends.htsim_rnic import FlowCompletion, RnicRunResult
from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement

_SEND_TAG_PATTERN = re.compile(r": send \d+b to \d+ tag (\d+)\b")
_DIMS = ModelDims(
    num_layers=2,
    hidden_size=1024,
    intermediate_size=4096,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=2,
)


def _record(step_index: int = 4) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=step_index * 1_000_000,
        scheduled=[
            ScheduledRequest(
                f"request-{step_index}",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=128,
            )
        ],
    )


def _config(workdir: Path, **overrides: object) -> HtsimStepSinkConfig:
    values = {
        "profile": "rnic-nn-fluid",
        "tp_ranks": (0, 1),
        "dims": _DIMS,
        "workdir": workdir,
    }
    values.update(overrides)
    return HtsimStepSinkConfig(**values)


def _active_goal_manifest(workdir: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(workdir.glob("step-*.artifact-*.goal"))
    )


def _install_completion_backend(monkeypatch):
    """Install a deterministic backend with an overlapping direct schedule."""

    calls = []
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def backend(config):
        calls.append(config)
        goal_text = Path(config.goal_bin).read_text()
        tags = tuple(int(tag) for tag in _SEND_TAG_PATTERN.findall(goal_text))
        is_cross_check = "atlahs-goal" in config.completion_csv.name
        step_match = re.search(r"step-(\d+)", config.completion_csv.name)
        step_index = int(step_match.group(1)) if step_match else 0
        if is_cross_check and step_index == 0:
            time.sleep(0.01)

        flows = []
        for index, tag in enumerate(tags):
            if is_cross_check:
                start_ps = index * 1_000
                completion_ps = start_ps + 100_000
            else:
                start_ps = 0
                completion_ps = 100_000
            flows.append(
                FlowCompletion(
                    profile=config.profile,
                    flow_id=index,
                    source=0,
                    destination=1,
                    tag=tag,
                    payload_bytes=64,
                    start_time_ps=start_ps,
                    completion_time_ps=completion_ps,
                    fct_ps=completion_ps - start_ps,
                )
            )
        return RnicRunResult(
            flows=flows,
            manifest=["stub completion backend"],
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", backend)
    return calls


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dependency_cross_check": "prefer-atlahs"}, "dependency_cross_check"),
        ({"dependency_cross_check_tolerance_ps": -1}, "nonnegative integer"),
        ({"dependency_cross_check_tolerance_ps": True}, "nonnegative integer"),
    ],
)
def test_cross_check_config_rejects_unknown_mode_and_invalid_tolerance(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **overrides)


def test_cross_check_is_opt_in_and_preserves_authoritative_outputs(
    tmp_path, monkeypatch
):
    calls = _install_completion_backend(monkeypatch)
    baseline_dir = tmp_path / "baseline"
    selected_dir = tmp_path / "selected"
    record = _record()

    baseline = HtsimStepSink(_config(baseline_dir))
    baseline_result = baseline(record)
    baseline_call_count = len(calls)
    selected = HtsimStepSink(
        _config(selected_dir, dependency_cross_check="atlahs-goal")
    )
    selected_result = selected(record)

    assert baseline_result == selected_result
    assert baseline.outcomes == selected.outcomes
    assert baseline.locality_outcomes == selected.locality_outcomes
    assert _active_goal_manifest(baseline_dir) == _active_goal_manifest(selected_dir)
    assert baseline.dependency_cross_check_reports == []
    assert not (baseline_dir / "cross-check").exists()
    assert len(calls) - baseline_call_count == baseline_call_count + 1
    assert (selected_dir / "cross-check").is_dir()


def test_selected_cross_check_reports_nonfatal_differences_and_tolerance(
    tmp_path, monkeypatch
):
    calls = _install_completion_backend(monkeypatch)
    exact = HtsimStepSink(
        _config(
            tmp_path / "exact",
            dependency_cross_check="atlahs-goal",
            dependency_cross_check_tolerance_ps=0,
        )
    )

    result = exact(_record())

    assert result is not None
    assert result.step_latency_ps == exact.outcomes[0].makespan_ps
    assert exact.locality_outcomes[0].ordering_authority == "execution-graph"
    assert len(exact.dependency_cross_check_reports) == 1
    report = exact.dependency_cross_check_reports[0]
    assert report.authority_mechanism == "execution-graph-projection"
    assert report.cross_check_mechanism == "atlahs-independent-goal"
    assert report.ordering_disagreement_count > 0
    assert any(item.disagreement for item in report.ordering_comparisons)
    assert report.phase_frontier_disagreement_count > 0
    assert any(
        item.evaluated
        and item.authority_gap_ps is not None
        and item.authority_gap_ps >= 0
        and item.cross_check_gap_ps is not None
        and item.cross_check_gap_ps < 0
        and item.disagreement
        for item in report.phase_frontier_comparisons
    )
    assert report.signed_completion_difference_ps < 0
    assert report.completion_tolerance_ps == 0
    assert report.completion_disagreement
    assert report.has_disagreement
    assert report.authority_completion_ps == result.step_latency_ps
    assert report.cross_check_artifact_name.startswith("cross-check/")
    assert report.cross_check_artifact_name.endswith(".atlahs-goal.goal")
    assert report.cross_check_artifact_name not in report.authority_artifact_names
    assert report.cross_check_artifact_bytes > 0
    assert len(report.cross_check_artifact_sha256) == 64
    assert all("artifact-" in name for name in report.authority_artifact_names)
    assert report.authority_quiescent and report.cross_check_quiescent
    assert report.authority_flow_count == exact.outcomes[0].num_flows
    assert report.cross_check_flow_count > 0
    assert sum("atlahs-goal" in call.completion_csv.name for call in calls) == 1

    tolerant = HtsimStepSink(
        _config(
            tmp_path / "tolerant",
            dependency_cross_check="atlahs-goal",
            dependency_cross_check_tolerance_ps=10**12,
        )
    )
    tolerant_result = tolerant(_record())
    tolerant_report = tolerant.dependency_cross_check_reports[0]

    assert tolerant_result == result
    assert not tolerant_report.completion_disagreement
    assert tolerant_report.ordering_disagreement_count > 0
    assert tolerant_report.phase_frontier_disagreement_count > 0
    assert tolerant_report.has_disagreement


def test_cross_check_rejects_local_nvlink_before_artifacts_or_backend(
    tmp_path, monkeypatch
):
    calls = _install_completion_backend(monkeypatch)
    workdir = tmp_path / "local"
    placement = PlacementManifest(
        ranks=[
            RankPlacement(0, "node-a", 0),
            RankPlacement(1, "node-a", 1),
        ]
    )
    selected = HtsimStepSink(
        _config(
            workdir,
            dependency_cross_check="atlahs-goal",
            placement_manifest=placement,
        )
    )

    with pytest.raises(ValueError, match="all-remote compatibility"):
        selected(_record())

    assert calls == []
    assert not list(workdir.rglob("*.goal"))
    assert not (workdir / "cross-check").exists()
    assert selected.outcomes == []
    assert selected.locality_outcomes == []
    assert selected.dependency_cross_check_reports == []


def test_persistent_cross_check_reports_publish_in_record_order(
    tmp_path, monkeypatch
):
    _install_completion_backend(monkeypatch)
    records = [_record(0), _record(1)]

    with HtsimPersistentStepSink(
        _config(
            tmp_path / "persistent",
            dependency_cross_check="atlahs-goal",
        ),
        max_workers=2,
    ) as sink:
        sink.prepare(records)
        assert sink.dependency_cross_check_reports == []
        assert sink.prepared_steps_remaining == 2
        results = [sink(record) for record in records]

    assert [result.step_index for result in results if result is not None] == [0, 1]
    assert [
        report.cross_check_artifact_name
        for report in sink.dependency_cross_check_reports
    ] == [
        "cross-check/step-000000.atlahs-goal.goal",
        "cross-check/step-000001.atlahs-goal.goal",
    ]
