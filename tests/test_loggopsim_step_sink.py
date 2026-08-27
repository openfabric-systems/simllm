"""The LogGOPSim ideal level reuses the checked step planning chain."""

from __future__ import annotations

from pathlib import Path

import pytest

import simllm.backends.loggopsim_step_sink as sink_module
from simllm.backends import (
    HtsimStepSink,
    HtsimStepSinkConfig,
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
    def convert(goal_path: Path) -> Path:
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
