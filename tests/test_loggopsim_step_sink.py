"""The LogGOPSim ideal level reuses the checked step planning chain."""

from __future__ import annotations

from pathlib import Path

import pytest

import simllm.backends.loggopsim_step_sink as sink_module
from simllm.backends import (
    HtsimStepSink,
    HtsimStepSinkConfig,
    LogGopsimFanInError,
    LogGopsimRunResult,
    LogGopsimStepSink,
    LogGopsimStepSinkConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement

_DIMS = ModelDims(
    num_layers=1,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    dtype_bytes=2,
)

_CLEAN_GOAL_SHA256 = {
    "a301c8b950a6d1514b98dc9442385fb94711706693b8a31fcc140bd43252ae0d",
    "fcef8e27444b2bcac146bcde20fbcdbc1d7f578835a107c22c4d7c61e16b15b2",
}


class _FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="declared")


def _record() -> StepRecord:
    return StepRecord(
        step_index=3,
        virtual_time_ps=7_000,
        scheduled=[
            ScheduledRequest(
                "request-0",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=8,
            )
        ],
    )


def _binary(path: Path) -> Path:
    path.write_bytes(b"pinned-loggopsim-test-binary")
    return path


def _stub_native(monkeypatch, *, makespan_ps: int = 123_000) -> None:
    def convert(goal_path: Path, bin_path=None, tool=None) -> Path:
        del bin_path
        assert tool is None or tool.name == "txt2bin"
        target = goal_path.with_suffix(".bin")
        target.write_bytes(b"binary:" + goal_path.read_bytes())
        return target

    def run(config, *, binary, timeout_s):
        assert config.byte_gap_ns_string == "0.02"
        assert binary.name == "LogGOPSim"
        assert timeout_s == 600
        return LogGopsimRunResult(
            rank_count=2,
            cpu_count=1,
            nic_count=1,
            max_finish_host=0,
            max_finish_ps=makespan_ps,
            host_finish_ps={0: makespan_ps},
            average_fct_ns=None,
            unmatched_queue_diagnostics=(),
        )

    monkeypatch.setattr(sink_module, "to_binary", convert)
    monkeypatch.setattr(sink_module, "run_loggopsim", run)


def _fan_in_goal(path: Path, *, serialize_receives: bool = False) -> Path:
    dependency = "r0op1 requires r0op0\n" if serialize_receives else ""
    path.write_text(
        "num_ranks 3\n"
        "rank 0 {\n"
        "r0op0: recv 64b from 1 tag 7\n"
        "r0op1: recv 64b from 2 tag 7\n"
        f"{dependency}"
        "}\n"
        "rank 1 {\n"
        "r1op0: send 64b to 0 tag 7\n"
        "}\n"
        "rank 2 {\n"
        "r2op0: send 64b to 0 tag 7\n"
        "}\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _direct_sink(tmp_path: Path, *, acknowledge_fan_in: bool = False):
    sink = LogGopsimStepSink(
        LogGopsimStepSinkConfig(
            tp_ranks=(0, 1),
            dims=_DIMS,
            workdir=tmp_path / "run",
            latency_ns=100,
            binary=_binary(tmp_path / "LogGOPSim"),
            provider=_FixedProvider(),
            acknowledge_fan_in=acknowledge_fan_in,
        )
    )
    return sink


def test_the_sink_prices_shared_goal_artifacts_and_records_exact_provenance(
    tmp_path, monkeypatch
):
    _stub_native(monkeypatch)
    sink = LogGopsimStepSink(
        LogGopsimStepSinkConfig(
            tp_ranks=(0, 1),
            dims=_DIMS,
            workdir=tmp_path / "run",
            latency_ns=100,
            binary=_binary(tmp_path / "LogGOPSim"),
            txt2bin=tmp_path / "txt2bin",
            provider=_FixedProvider(),
        )
    )
    result = sink(_record())
    assert result is not None
    assert result.step_latency_ps == 2_000 + 2 * 123_000
    assert result.completed_at_ps == 7_000 + result.step_latency_ps

    provenance = sink.provenance
    assert len(provenance.invocations) == 2
    assert provenance.parameters.exact_g_string == "0.02"
    assert {
        parameter["evidence_source"]
        for parameter in provenance.parameters.to_json().values()
    } == {"DECLARED"}
    for invocation in provenance.invocations:
        assert invocation.exact_g_string == "0.02"
        assert invocation.argv[invocation.argv.index("-G") + 1] == "0.02"
        assert invocation.max_finish_ps == 123_000
        assert len(invocation.goal_sha256) == 64
        assert len(invocation.goal_binary_sha256) == 64
        assert invocation.goal_sha256 in _CLEAN_GOAL_SHA256
        assert invocation.fan_in.to_json() == {
            "schema": "simllm-loggopsim-fan-in-envelope-v1",
            "fan_in_detected": False,
            "acknowledged": False,
            "mechanism": "receiver per-byte gap unmodeled",
            "frozen_cell_error": "about 8x optimistic",
            "study": "examples/frontier_ladder_v1/RESULTS.md",
            "destinations": [],
        }


def test_sink_refuses_unacknowledged_receiver_fan_in_before_native_work(
    tmp_path, monkeypatch
):
    _stub_native(monkeypatch)
    sink = _direct_sink(tmp_path)
    goal = _fan_in_goal(tmp_path / "incast.goal")

    with pytest.raises(LogGopsimFanInError) as error:
        sink._sink._run_goal(None, goal, tmp_path / "unused.csv")

    diagnostic = str(error.value)
    assert "receiver per-byte gap is unmodeled" in diagnostic
    assert "about 8x optimistic" in diagnostic
    assert "examples/frontier_ladder_v1/RESULTS.md" in diagnostic
    assert "acknowledge_fan_in=True" in diagnostic
    assert not goal.with_suffix(".bin").exists()
    assert sink.provenance.invocations == ()


def test_sink_acknowledges_receiver_fan_in_and_stamps_the_run(tmp_path, monkeypatch):
    _stub_native(monkeypatch)
    sink = _direct_sink(tmp_path, acknowledge_fan_in=True)
    goal = _fan_in_goal(tmp_path / "acknowledged-incast.goal")

    result = sink._sink._run_goal(None, goal, tmp_path / "unused.csv")

    assert result.max_finish_ps == 123_000
    stamp = sink.provenance.invocations[0].fan_in.to_json()
    assert stamp["fan_in_detected"] is True
    assert stamp["acknowledged"] is True
    assert stamp["destinations"] == [
        {"receiver_rank": 0, "source_ranks": [1, 2]}
    ]


def test_sink_accepts_explicitly_serialized_receives_without_acknowledgment(
    tmp_path, monkeypatch
):
    _stub_native(monkeypatch)
    sink = _direct_sink(tmp_path)
    goal = _fan_in_goal(tmp_path / "serialized.goal", serialize_receives=True)
    before = goal.read_bytes()

    result = sink._sink._run_goal(None, goal, tmp_path / "unused.csv")

    assert result.max_finish_ps == 123_000
    assert goal.read_bytes() == before
    stamp = sink.provenance.invocations[0].fan_in
    assert stamp.fan_in_detected is False
    assert stamp.acknowledged is False
    assert stamp.destinations == ()


def test_intra_node_service_is_identical_to_the_existing_analytic_path(tmp_path):
    manifest = PlacementManifest(
        ranks=[
            RankPlacement(0, "node-a", 0),
            RankPlacement(1, "node-a", 1),
        ]
    )
    common = {
        "tp_ranks": (0, 1),
        "dims": _DIMS,
        "provider": _FixedProvider(),
        "placement_manifest": manifest,
    }
    ideal = LogGopsimStepSink(
        LogGopsimStepSinkConfig(
            **common,
            workdir=tmp_path / "ideal",
            latency_ns=100,
            binary=_binary(tmp_path / "LogGOPSim"),
        )
    )
    existing = HtsimStepSink(
        HtsimStepSinkConfig(
            **common,
            profile="rnic-nn-fluid",
            workdir=tmp_path / "existing",
        )
    )
    ideal_result = ideal(_record())
    existing_result = existing(_record())
    assert ideal_result == existing_result
    assert ideal.locality_outcomes == existing.locality_outcomes
    assert ideal.provenance.invocations == ()


def test_eager_only_guard_has_a_payload_mutation_negative_control(
    tmp_path, monkeypatch
):
    _stub_native(monkeypatch)
    sink = LogGopsimStepSink(
        LogGopsimStepSinkConfig(
            tp_ranks=(0, 1),
            dims=_DIMS,
            workdir=tmp_path / "mutant",
            latency_ns=100,
            rendezvous_threshold_bytes=1,
            binary=_binary(tmp_path / "LogGOPSim"),
            provider=_FixedProvider(),
        )
    )
    with pytest.raises(ValueError, match="every rendered payload to be eager"):
        sink(_record())
    assert sink.provenance.invocations == ()
