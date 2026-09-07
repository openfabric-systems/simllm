"""Packet visits conserve runtime time and preserve accepted replay artifacts."""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
)
from simllm.backends.packet_breakdown import (
    PacketStepBreakdown,
    packet_step_from_json,
    packet_step_to_json,
)
from simllm.compute import ComputeProvider, DurationEstimate, HostInitiationModel, ModelDims
from simllm.core import (
    EventPhase,
    RequestPhase,
    ResourceKind,
    ScheduledRequest,
    StepRecord,
    completion_event_from_json,
    completion_event_to_json,
    step_record_from_json,
    step_result_to_json,
)
from simllm.placement import declared_manifest

FIXTURES = Path(__file__).resolve().parents[1] / "examples/packet_breakdown_v1/fixtures.json"
SPEC = importlib.util.spec_from_file_location("packet_study", FIXTURES.with_name("run_study.py"))
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)
artifact_digest, cases, digest = STUDY.artifact_digest, STUDY.cases, STUDY.digest
exact_snapshot, replay, unpack_runs = STUDY.exact_snapshot, STUDY.replay, STUDY.unpack_runs
DIMS = ModelDims(num_layers=1, hidden_size=8, intermediate_size=16, num_heads=2,
                 num_kv_heads=2, head_size=4, vocab_size=16, dtype_bytes=2)


class Fixed(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=10_000, bound="measured")


def config(tmp_path, *, profile="rnic-nn", **kwargs):
    return HtsimStepSinkConfig(profile=profile, tp_ranks=(0, 1), dims=DIMS,
                               workdir=tmp_path, provider=Fixed(), **kwargs)


def record(index=0, release=0, *, sampled=2):
    return StepRecord(index, release, [
        ScheduledRequest("a", RequestPhase.DECODE, 1, context_length=8),
        ScheduledRequest("b", RequestPhase.PREFILL, 3, context_length=8),
    ], num_sampled=sampled)


def stub(sink, *, start=37, finish=10_501, job=11_000, quiescent=True):
    def run(plan, goal, csv):
        return SimpleNamespace(
            flows=[SimpleNamespace(start_time_ps=start, completion_time_ps=finish,
                                   fct_ps=finish - start)],
            quiescent=quiescent, job_completion_time_ps=lambda: job)
    sink._run_goal = run
    return sink


def execute(tmp_path, **kwargs):
    sink = stub(HtsimStepSink(config(tmp_path, emit_packet_breakdown=True, **kwargs)))
    result = sink(record())
    return sink, result, sink.packet_breakdowns[-1]


def test_real_visit_extrema_and_rounding_drive_five_components(tmp_path):
    sink, result, projection = execute(tmp_path)
    b = projection.breakdown
    assert (b.launch_queue_ps, b.device_queue_ps, b.service_ps,
            b.completion_delivery_ps, b.external_dependency_ps) == (
                0, 74, 20_928, 998, 10_000)
    assert b.operation_latency_ps == result.step_latency_ps == 32_000
    assert b.critical_path_queue_ps == 74
    assert attribute_step_detail(result, sink.locality_outcomes[-1],
                                 packet_breakdown=projection) == attribute_step_detail(
                                     result, sink.locality_outcomes[-1])
    report = projection.runtime_report
    assert report.visits == projection.visits
    assert report.sum_visit_wait_ps == sum(v.queue_wait_ps for v in projection.visits)
    assert sum(o.breakdown.operation_latency_ps for o in report.operations) == 32_000
    assert len(report.operations) == sink.locality_outcomes[-1].artifact_count
    assert all(s.attribution.total_ps == s.breakdown.operation_latency_ps
               for s in projection.segments)
    for visit in projection.visits:
        assert (visit.submitted_at_ps <= visit.eligible_at_ps <= visit.started_at_ps
                <= visit.finished_at_ps <= visit.completed_at_ps)


def test_events_expose_all_five_timestamps_and_keep_visit_identity(tmp_path):
    _, result, projection = execute(tmp_path)
    execution = projection.execution_result
    assert execution.completed_at_ps == result.completed_at_ps
    assert [e.timestamp_ps for e in execution.events] == sorted(e.timestamp_ps for e in execution.events)
    assert len(execution.events) == 5 * len(projection.visits)
    for visit in projection.visits:
        events = [e for e in execution.events if e.subject_object_id == visit.subject_object_id]
        assert {e.phase: e.timestamp_ps for e in events} == {
            EventPhase.SUBMITTED: visit.submitted_at_ps,
            EventPhase.QUEUED: visit.eligible_at_ps,
            EventPhase.STARTED: visit.started_at_ps,
            EventPhase.PROGRESS: visit.finished_at_ps,
            EventPhase.COMPLETED: visit.completed_at_ps,
        }
        assert all(completion_event_from_json(completion_event_to_json(e)) == e for e in events)


def test_registration_and_host_floor_are_partitioned_once(tmp_path):
    off, old, _ = execute(tmp_path / "off")
    sink, result, projection = execute(
        tmp_path / "on", host_model=HostInitiationModel(4_000, "gin"),
        collective_registration="nccl-channel-registration-v1")
    charge = sink.collective_registration_outcomes[-1].charged_ps
    assert charge > 0
    assert result.step_latency_ps == old.step_latency_ps + 4_000 + charge
    assert projection.breakdown.launch_queue_ps == 4_000 + charge
    assert projection.breakdown.external_dependency_ps == 10_000
    assert projection.detail.media.kernel_ps == off.packet_breakdowns[-1].detail.media.kernel_ps + 4_000
    assert projection.detail.media.collective_registration_ps == charge
    second = sink(record(1, result.completed_at_ps))
    assert sink.packet_breakdowns[-1].breakdown.launch_queue_ps == 4_000
    assert second.step_latency_ps == old.step_latency_ps + 4_000


@pytest.mark.parametrize("nodes", [1, 2])
def test_local_and_mixed_media_preserve_masked_work(tmp_path, nodes):
    cfg = config(tmp_path, emit_packet_breakdown=True,
                 placement_manifest=declared_manifest(tp=4, nodes=nodes, gpus_per_node=4 // nodes))
    cfg.tp_ranks = tuple(range(4))
    sink = stub(HtsimStepSink(cfg))
    result = sink(record())
    projection = sink.packet_breakdowns[-1]
    assert projection.detail == attribute_step_detail(result, sink.locality_outcomes[-1])
    if nodes == 1:
        assert projection.detail.media.fabric_ps == 0
        assert projection.detail.media.nvlink_ps > 0
        assert any(v.resource.kind == ResourceKind.NVLINK for v in projection.visits)
    else:
        assert projection.detail.media.fabric_ps > 0
        assert projection.detail.masked.nvlink_ps > 0


def test_pending_requests_and_scheduler_gaps_conserve_both_intervals(tmp_path):
    sink = stub(HtsimStepSink(config(tmp_path, emit_packet_breakdown=True)))
    enabled, old = HtsimRequestMetricReducer({"a": 0, "b": 100}), HtsimRequestMetricReducer({"a": 0, "b": 100})
    now = 200
    for index, sampled in enumerate((1, 1, 2, 2)):
        rec = record(index, now, sampled=sampled)
        result = sink(rec)
        locality, projection = sink.locality_outcomes[-1], sink.packet_breakdowns[-1]
        assert enabled.consume(rec, result, locality, packet_breakdown=projection) == old.consume(rec, result, locality)
        now = result.completed_at_ps + 53
    assert enabled.totals() == old.totals()
    for row in enabled.totals():
        ttft, decode = enabled.critical_path_breakdowns()[row.request_id]
        assert ttft.operation_latency_ps == row.ttft_ps
        assert decode.operation_latency_ps == row.decode_attribution.total_ps
    assert old.critical_path_breakdowns() == {}


def test_absent_field_preserves_exact_old_serialization(tmp_path):
    _, result, projection = execute(tmp_path)
    original = step_result_to_json(result)
    assert packet_step_to_json(result) == original
    decoded, absent = packet_step_from_json(json.loads(json.dumps(original)))
    assert decoded == result and absent is None
    assert packet_step_to_json(decoded, absent) == original
    with pytest.raises(ValueError):
        packet_step_from_json({**original, "critical_path_breakdown": None})
    encoded = packet_step_to_json(result, projection)
    assert packet_step_from_json(json.loads(json.dumps(encoded))) == (result, projection)
    assert asdict(result) == asdict(decoded)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema="future"),
    lambda p: p.update(unrecognized=1),
    lambda p: p["breakdown"].update(service_ps=True),
    lambda p: p["breakdown"].update(operation_latency_ps=0),
    lambda p: p.update(masked_fabric_ps=-1),
    lambda p: p["visits"][0].update(submitted_at_ps=False),
    lambda p: p["visits"][0].update(eligible_at_ps=-1),
    lambda p: p["visits"][1].update(started_at_ps=100_000),
    lambda p: p["visits"][1].update(stage="unknown"),
    lambda p: p["visits"][1].update(stage="fabric_ps"),
    lambda p: p["visits"][1]["resource"].update(kind="unknown"),
    lambda p: p["visits"][1].update(extra=0),
    lambda p: p["visits"][1].update(operation_id="alien"),
    lambda p: p["visits"][1].update(subject_object_id=p["visits"][0]["subject_object_id"]),
])
def test_strict_wire_rejects_malformed_projection(tmp_path, mutation):
    _, result, projection = execute(tmp_path)
    payload = packet_step_to_json(result, projection)
    mutation(payload["critical_path_breakdown"])
    with pytest.raises((ValueError, TypeError)):
        packet_step_from_json(payload)


def test_mismatched_result_and_locality_fail_closed(tmp_path):
    sink, result, projection = execute(tmp_path)
    with pytest.raises(ValueError, match="StepResult"):
        packet_step_to_json(replace(result, step_index=8), projection)
    changed = replace(sink.locality_outcomes[-1], nvlink_service_ps=1)
    with pytest.raises(ValueError):
        attribute_step_detail(result, changed, packet_breakdown=projection)
    with pytest.raises(ValueError):
        PacketStepBreakdown(0, tuple(reversed(projection.visits)))


@pytest.mark.parametrize("persistent", [False, True])
def test_disabled_selection_changes_no_accepted_outcome_or_prepared_batch(tmp_path, persistent):
    sinks, outputs = [], []
    for name, selected in (("default", None), ("off", False), ("on", True)):
        cfg = config(tmp_path / name, **({} if selected is None else {"emit_packet_breakdown": selected}))
        sink = stub(HtsimPersistentStepSink(cfg, max_workers=2) if persistent else HtsimStepSink(cfg))
        records = [record(0, 0), record(1, 32_000)]
        try:
            if persistent:
                sink.prepare(records)
                assert sink.outcomes == sink.packet_breakdowns == []
            outputs.append([sink(r) for r in records])
            sinks.append(sink)
        finally:
            if persistent:
                sink.close()
    assert outputs[0] == outputs[1] == outputs[2]
    assert sinks[0].locality_outcomes == sinks[1].locality_outcomes == sinks[2].locality_outcomes
    assert sinks[0].packet_breakdowns == sinks[1].packet_breakdowns == []
    assert len(sinks[2].packet_breakdowns) == 2
    assert len({artifact_digest(s.config.workdir, {".goal"}) for s in sinks}) == 1


def test_projection_guard_leaves_failed_preparation_unpublished(tmp_path):
    sink = stub(HtsimPersistentStepSink(config(tmp_path, emit_packet_breakdown=True), max_workers=2), quiescent=False)
    with sink, pytest.raises(ValueError, match="quiescent"):
        sink.prepare([record()])
    assert sink.outcomes == sink.packet_breakdowns == []
    assert sink.prepared_steps_remaining == 0


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_every_retained_study_fixture_reaches_identical_shares(tmp_path, case):
    retained = json.loads(FIXTURES.read_bytes())
    fixture = next(f for f in retained["cells"] if f["case"] == case)
    records = [step_record_from_json(r) for r in fixture["records"]]
    runs = unpack_runs(fixture["runs"])
    first = None
    for enabled, persistent in ((None, False), (False, True), (True, False), (True, True)):
        directory = tmp_path / f"{enabled}-{persistent}"
        sink, results, metrics, reducer = replay(case, directory, copy.deepcopy(records), runs,
                                                 enabled=enabled, persistent=persistent)
        assert digest(exact_snapshot(sink, results)) == fixture["snapshot_sha256"]
        assert artifact_digest(directory, {".goal"}) == fixture["goal_sha256"]
        if first is None:
            first = metrics, reducer.totals()
        assert (metrics, reducer.totals()) == first
        if enabled:
            for result, projection in zip(results, sink.packet_breakdowns, strict=True):
                assert projection.breakdown.operation_latency_ps == result.step_latency_ps
                assert packet_step_from_json(packet_step_to_json(result, projection)) == (result, projection)
            for row in reducer.totals():
                ttft, decode = reducer.critical_path_breakdowns()[row.request_id]
                assert ttft.operation_latency_ps == row.ttft_ps
                assert decode.operation_latency_ps == row.decode_attribution.total_ps


def test_retained_input_digests_accept_raw_or_lf_bytes():
    retained = json.loads(FIXTURES.read_bytes())
    root = FIXTURES.parents[2]
    for name, expected in retained["inputs"].items():
        raw = (root / name).read_bytes()
        assert expected in {digest(raw), digest(raw.replace(b"\r\n", b"\n"))}


@pytest.mark.parametrize("mode", ["local", "fabric", "tie"])
def test_mixed_artifact_maximum_retains_its_owner(tmp_path, mode):
    cfg = config(tmp_path, emit_packet_breakdown=True,
                 placement_manifest=declared_manifest(tp=4, nodes=2, gpus_per_node=2))
    cfg.tp_ranks = (0, 1, 2, 3)
    sink = HtsimStepSink(cfg)

    def run(plan, goal, csv):
        artifact = next(a for a in plan.artifacts if a.goal_path == goal)
        local = artifact.local_service_ps
        duration = max(1, local + {"local": -1, "fabric": 1, "tie": 0}[mode])
        return SimpleNamespace(
            flows=[SimpleNamespace(start_time_ps=0, completion_time_ps=duration,
                                   fct_ps=duration)], quiescent=True,
            job_completion_time_ps=lambda: duration)

    sink._run_goal = run
    result = sink(record())
    projection = sink.packet_breakdowns[-1]
    assert projection.detail == attribute_step_detail(result, sink.locality_outcomes[-1])
    if mode == "local":
        assert projection.detail.masked.fabric_ps > 0
    elif mode == "fabric":
        assert projection.detail.masked.nvlink_ps > 0
    else:
        assert projection.detail.media.co_critical_ps > 0


def test_runtime_report_scalars_reproduce_the_core_segment_contract(tmp_path):
    from simllm.core.completion import _validate_scalar_projection

    _, _, projection = execute(tmp_path)
    by_id = {o.operation_id: o for o in projection.runtime_report.operations}
    for operation in by_id.values():
        _validate_scalar_projection(operation, by_id, 0)


def test_semantic_base_stays_named_in_completion_delivery(tmp_path):
    from simllm.traffic import B200_NCCL_2_27_LOCAL_PROFILE

    sink, result, projection = execute(
        tmp_path, profile="rnic-nn-fluid", collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE)
    base = sum(sink.locality_outcomes[-1].base_phase_latency_ps)
    assert base > 0
    assert projection.breakdown.completion_delivery_ps == base + 998
    assert projection.detail.media.collective_base_ps == base
    assert projection.detail == attribute_step_detail(result, sink.locality_outcomes[-1])


def test_stateful_multi_artifact_guard_remains_active_when_opted_in(tmp_path):
    cfg = config(tmp_path, emit_packet_breakdown=True)
    cfg.profile = "rnic-cn"
    sink = HtsimStepSink(cfg)
    with pytest.raises(RuntimeError, match="resets simulator state"):
        sink(record())
    assert sink.outcomes == sink.packet_breakdowns == []


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_opt_in_requires_an_explicit_boolean(tmp_path, value):
    with pytest.raises(TypeError, match="boolean"):
        config(tmp_path, emit_packet_breakdown=value)
