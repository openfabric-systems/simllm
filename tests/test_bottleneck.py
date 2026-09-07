"""Selected-path rankings preserve timing and reject misleading wire evidence."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
)
from simllm.backends.device_step_sink import DeviceRuntimeStepSink
from simllm.backends.step_lowerer import SerialStepLowererConfig
from simllm.compute import GPU_ENVELOPES, ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CompletionReducer,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    step_result_to_json,
)
from simllm.core.bottleneck import (
    BottleneckHistory,
    BottleneckRanking,
    BottleneckReport,
    ClassShare,
    FlowTail,
    KernelEvidence,
    bottleneck_report_from_json,
    bottleneck_report_to_json,
    bottleneck_step_from_json,
    bottleneck_step_to_json,
    classify_runtime,
    combine,
    ranking,
)

DIMS = ModelDims(num_layers=1, hidden_size=8, intermediate_size=16, num_heads=2,
                 num_kv_heads=2, head_size=4, vocab_size=16, dtype_bytes=2)


class Fixed(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=10_000, bound="measured")


def record(index=0, release=0, *, sampled=None):
    return StepRecord(index, release, [
        ScheduledRequest("a", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE, 1, context_length=8),
        ScheduledRequest("b", RequestPhase.DECODE, 1, context_length=8),
    ], sampled_request_ids=sampled)


def config(path, **kwargs):
    return HtsimStepSinkConfig(profile="rnic-nn", tp_ranks=(0, 1), dims=DIMS,
                               workdir=path, provider=Fixed(), **kwargs)


def stub(sink):
    def run(plan, goal, csv):
        csv.write_bytes(b"retained-packet-rows\n")
        return SimpleNamespace(flows=[
            SimpleNamespace(start_time_ps=37, completion_time_ps=10_501, fct_ps=10_464),
            SimpleNamespace(start_time_ps=100, completion_time_ps=10_501, fct_ps=10_401),
        ], quiescent=True, job_completion_time_ps=lambda: 11_000)
    sink._run_goal = run
    return sink


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_packet_identity_default_false_true_and_prepared(tmp_path):
    results, snapshots = [], []
    for mode in (None, False, True, "prepared"):
        directory = tmp_path / str(mode)
        cfg = config(directory, **({} if mode is None else {"emit_bottleneck_report": bool(mode)}))
        if mode == "prepared":
            sink = stub(HtsimPersistentStepSink(cfg, max_workers=2))
            sink.prepare([record()])
            assert sink.bottleneck_reports == []
        else:
            sink = stub(HtsimStepSink(cfg))
        result = sink(record())
        results.append(wire(step_result_to_json(result)))
        snapshots.append({p.name: p.read_bytes() for p in directory.iterdir()})
        if mode:
            report = sink.bottleneck_reports[-1]
            assert report.step.span_ps == result.step_latency_ps == 32_000
            assert report.step.classes[0] == ClassShare("fabric", 2, 22_000)
            assert all(f.tail_share == Fraction(63, 10_464) for f in report.step.flow_tails)
            assert all(k.achieved_fraction is None for k in report.step.kernels)
            payload = bottleneck_step_to_json(result, report)
            decoded, projected = bottleneck_step_from_json(json.loads(wire(payload)))
            assert decoded == result and projected == report
            del payload["bottleneck_report"]
            assert wire(payload) == results[-1]
        else:
            assert sink.bottleneck_reports == [] and sink.packet_breakdowns == []
            assert bottleneck_step_to_json(result) == step_result_to_json(result)
            assert bottleneck_step_from_json(step_result_to_json(result)) == (result, None)
        if mode == "prepared":
            sink.close()
    assert len(set(results)) == 1
    assert all(snapshot == snapshots[0] for snapshot in snapshots)


def test_packet_request_chunks_gaps_and_fractional_average(tmp_path):
    sink = stub(HtsimStepSink(config(tmp_path, emit_bottleneck_report=True)))
    reducer = HtsimRequestMetricReducer({"a": 0, "b": 0})
    release = 11
    for index, (gap, sampled) in enumerate(((0, []), (13, ["a", "b"]), (17, ["a", "b"]), (18, ["a", "b"]))):
        release += gap
        source = record(index, release, sampled=sampled)
        result = sink(source)
        metrics = reducer.consume(source, result, sink.locality_outcomes[-1],
                                  packet_breakdown=sink.packet_breakdowns[-1],
                                  bottleneck_report=sink.bottleneck_reports[-1])
        report = reducer.bottleneck_reports[-1]
        report.validate_result(replace(result, request_metrics=metrics))
        if index == 0:
            assert report.requests == ()
        elif index == 1:
            assert all(r.ttft.span_ps == 64_024 for r in report.requests)
            assert all(r.ttft.share("batching-queue") == Fraction(24, 64_024) for r in report.requests)
        elif index == 3:
            assert all(r.tpot.latency_ps == Fraction(64_035, 2) for r in report.requests)
        release = result.completed_at_ps
    payload = bottleneck_step_to_json(replace(result, request_metrics=metrics), report)
    for row in payload["bottleneck_report"]["requests"]:
        row["tpot"]["divisor"] *= 2
        row["tpot"]["span_ps"] *= 2
        for cls in row["tpot"]["classes"]:
            cls["span_ps"] *= 2
    with pytest.raises(ValueError, match="divisor"):
        bottleneck_step_from_json(payload)
    with pytest.raises(ValueError, match="mix"):
        reducer.consume(record(4, release), sink(record(4, release)), sink.locality_outcomes[-1])


def operation(name, rank, duration, request):
    return ExecutionOperation(name, rank, f"queue:{rank}", ComputeWork(
        "kernel", flops=1000, hbm_bytes=1000, nominal_duration_ps=duration),
        correlation=OperationCorrelation(request_ids=(request,)))


def test_coarse_unequal_endpoints_and_queue_selection():
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile(launch_service_ps=7, completion_delivery_ps=3))
    reducer = CompletionReducer(VirtualClock(), emit_bottleneck_report=True)
    for index in range(3):
        release = reducer.clock.now_ps
        source = record(index, release)
        graph = ExecutionGraph(f"graph:{index}", index, release,
                               (operation("fast", 0, 30, "a"), operation("slow", 1, 101, "b")),
                               completion_operation_ids=("fast", "slow"))
        execution = runtime.execute(graph)
        report = runtime.last_report
        segments = classify_runtime(graph, report, runtime.selected_critical_visits, GPU_ENVELOPES["b100"])
        result = reducer.reduce(source, graph, execution, report, bottleneck_segments=segments)
        bottleneck = reducer.bottleneck_reports[-1]
        assert bottleneck.step.span_ps == result.step_latency_ps
        assert bottleneck.requests[0].ttft.span_ps < bottleneck.requests[1].ttft.span_ps
        assert bottleneck.step.share("host-launch") > 0
        if index:
            assert bottleneck.requests[0].tpot.share("batching-queue") > 0
        assert bottleneck_report_from_json(bottleneck_report_to_json(bottleneck)) == bottleneck


def test_device_sink_opt_in_and_kernel_identity(tmp_path):
    baseline = []
    for enabled in (False, True):
        sink = DeviceRuntimeStepSink(SerialStepLowererConfig(
            tp_ranks=(0, 1), dims=DIMS, provider=Fixed()), emit_bottleneck_report=enabled)
        sink.bind_clock(VirtualClock())
        result = sink(record(), None)
        baseline.append(wire(step_result_to_json(result)))
        if enabled:
            report = sink.bottleneck_reports[-1]
            assert report.step.kernels
            assert any(c.class_name == "intra-node-collective" and c.participant_width == 2 for c in report.step.classes)
            work = next(o.work for o in sink.outcomes[-1].graph.operations if isinstance(o.work, ComputeWork))
            assert report.step.kernels[0].kernel_id == work.kernel
            assert report.step.kernels[0].config == work.config
        else:
            assert sink.bottleneck_reports == []
    assert baseline[0] == baseline[1]


def example_report():
    k = KernelEvidence("gemm", (("m", 8),), 1000, 1000, 1e12, 1e12, "test-gpu",
                       "constants/boosted/cell", 2000, "retained-standalone-microbenchmark", "void")
    r = ranking((("compute-bound", None, 23), ("fabric", 8, 50)), kernels=(k,),
                flow_tails=(FlowTail("execution", "artifact", 3, 10, 20),))
    return BottleneckReport(0, r)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema="future"),
    lambda p: p.update(extra=1),
    lambda p: p.update(step_index=True),
    lambda p: p["step"].update(span_ps=True),
    lambda p: p["step"].update(span_ps=74),
    lambda p: p["step"].update(divisor=0),
    lambda p: p["step"]["classes"].reverse(),
    lambda p: p["step"]["classes"][0].update(class_name="network-maybe"),
    lambda p: p["step"]["classes"][0].update(participant_width=True),
    lambda p: p["step"]["classes"][0].update(share={"numerator": 100, "denominator": 146}),
    lambda p: p["step"]["classes"][0].update(span_ps=-1),
    lambda p: p["step"]["classes"].append(p["step"]["classes"][0]),
    lambda p: p["step"].update(kernels=[]),
    lambda p: p["step"]["kernels"][0].update(peak_flops_s=float("nan")),
    lambda p: p["step"]["kernels"][0].update(hbm_bytes_s=True),
    lambda p: p["step"]["kernels"][0].update(bound_class="hbm-bound"),
    lambda p: p["step"]["kernels"][0].update(achieved_fraction=True),
    lambda p: p["step"]["kernels"][0].update(achieved_fraction=0.9),
    lambda p: p["step"]["kernels"][0].update(source_state="accepted"),
    lambda p: p["step"]["kernels"][0].update(measured_time_ps=False),
    lambda p: p["step"]["kernels"][0].update(ledger_cell_id=None),
    lambda p: p["step"]["kernels"][0]["config"].append(["m", 8]),
    lambda p: p["step"]["flow_tails"][0].update(max_fct_ps=9),
    lambda p: p["step"]["flow_tails"][0].update(flow_count=0),
    lambda p: p["step"]["flow_tails"][0].update(tail_share={"numerator": 0, "denominator": 1}),
    lambda p: p["step"]["flow_tails"].append(p["step"]["flow_tails"][0]),
])
def test_strict_nested_codec_rejects_corruption(mutate):
    payload = bottleneck_report_to_json(example_report())
    mutate(payload)
    with pytest.raises((ValueError, TypeError)):
        bottleneck_report_from_json(payload)


def test_exact_ridge_void_state_and_missing_measurement():
    source = example_report()
    k = source.step.kernels[0]
    assert k.bound_class == "compute-bound" and k.achieved_fraction == 0.5
    assert bottleneck_report_from_json(json.loads(wire(bottleneck_report_to_json(source)))) == source
    assert replace(k, flops=999).bound_class == "hbm-bound"
    assert replace(k, flops=1001).bound_class == "compute-bound"
    missing = KernelEvidence("kernel", (), 0, 0, 1, 1, "test-gpu")
    assert missing.bound_class == "unclassified-kernel" and missing.achieved_fraction is None
    assert ranking().latency_ps == 0 and ranking().share("fabric") == 0
    with pytest.raises(ValueError, match="undivided"):
        combine(replace(ranking(), divisor=2))


def test_request_wire_identity_and_missing_optional_field(tmp_path):
    sink = stub(HtsimStepSink(config(tmp_path, emit_bottleneck_report=True)))
    result = sink(record())
    payload = bottleneck_step_to_json(result)
    payload["bottleneck_report"] = None
    with pytest.raises(ValueError):
        bottleneck_step_from_json(payload)
    with pytest.raises(ValueError, match="disagrees"):
        bottleneck_step_to_json(result, replace(sink.bottleneck_reports[-1], step_index=9))


def test_failed_history_validation_does_not_publish():
    history = BottleneckHistory()
    from simllm.core import StepResult
    source = record()
    with pytest.raises(ValueError):
        history.consume(source, StepResult(9, 10, 10), ranking((("control", None, 10),)),
                        {"a": ranking(), "b": ranking()}, {"a": 0, "b": 0})
    assert history.reports == [] and history._states == {}


def test_noncritical_parallel_work_is_not_added():
    runtime = CoarseDeviceRuntime()
    source = record()
    graph = ExecutionGraph("parallel", 0, 0,
                           (operation("fast", 0, 10, "a"), operation("slow", 1, 100, "b")),
                           completion_operation_ids=("fast", "slow"))
    result = runtime.execute(graph)
    report = runtime.last_report
    segments = classify_runtime(graph, report, runtime.selected_critical_visits, GPU_ENVELOPES["b100"])
    selected = combine(*(segments[k] for k in report.realized_critical_path_segments))
    assert selected.span_ps == result.completed_at_ps == 100
    assert sum(s.span_ps for s in segments.values()) == 110
    reducer = CompletionReducer(VirtualClock(), emit_bottleneck_report=True)
    output = reducer.reduce(source, graph, result, report, bottleneck_segments=segments)
    before = copy.deepcopy(asdict(output))
    _ = bottleneck_step_to_json(output, reducer.bottleneck_reports[-1])
    assert asdict(output) == before


def test_in_memory_conservation_and_duplicates():
    with pytest.raises(ValueError, match="conserve"):
        BottleneckRanking(2, classes=(ClassShare("control", None, 1),))
    with pytest.raises(ValueError, match="duplicate"):
        BottleneckRanking(2, classes=(ClassShare("control", None, 1),) * 2)


def test_measurement_join_uses_full_calibration_key_and_work():
    from simllm.core.bottleneck import join_kernel_cell
    k = example_report().step.kernels[0]
    key = k.kernel_id, k.config, k.gpu_name
    absent = replace(k, ledger_cell_id=None, measured_time_ps=None,
                     evidence_class="absent-by-design", source_state="absent-by-design")
    assert join_kernel_cell(absent, {key: k}) == k
    assert join_kernel_cell(absent, {(k.kernel_id, k.config, "another-gpu"): k}) == absent
    with pytest.raises(ValueError, match="identity"):
        join_kernel_cell(absent, {key: replace(k, gpu_name="another-gpu")})
    with pytest.raises(ValueError, match="work"):
        join_kernel_cell(absent, {key: replace(k, flops=k.flops + 1)})


def test_selected_path_projection_is_read_only_and_loss_checked():
    runtime = CoarseDeviceRuntime()
    graph = ExecutionGraph("selected", 0, 0, (operation("k", 0, 10, "a"),),
                           completion_operation_ids=("k",))
    runtime.execute(graph)
    paths = runtime.selected_critical_visits
    paths.clear()
    assert runtime.selected_critical_visits
    with pytest.raises(ValueError, match="retain"):
        classify_runtime(graph, runtime.last_report, paths, GPU_ENVELOPES["b100"])
    paths = runtime.selected_critical_visits
    paths["k", 0] = ()
    with pytest.raises(ValueError, match="nonempty"):
        classify_runtime(graph, runtime.last_report, paths, GPU_ENVELOPES["b100"])


def test_host_launch_can_rank_first_without_changing_kernel_service():
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile(launch_service_ps=100))
    graph = ExecutionGraph("host", 0, 0, (operation("k", 0, 10, "a"),),
                           completion_operation_ids=("k",))
    result = runtime.execute(graph)
    by_segment = classify_runtime(graph, runtime.last_report, runtime.selected_critical_visits, GPU_ENVELOPES["b100"])
    row = by_segment["k", 0]
    assert row.classes[0] == ClassShare("host-launch", None, 100)
    assert row.span_ps == result.completed_at_ps == 110


def test_live_packet_measurement_join_preserves_void_and_timing(tmp_path):
    from simllm.compute import step_kernel
    cfg = config(tmp_path / "joined", emit_bottleneck_report=True)
    kernel = step_kernel(cfg.dims, record(), num_sampled=2)
    evidence = replace(KernelEvidence.from_kernel(kernel, cfg.gpu),
                       ledger_cell_id="synthetic-fixture/kernel", measured_time_ps=10_000,
                       evidence_class="synthetic-fixture", source_state="void")
    cfg.bottleneck_kernel_cells = {(kernel.name, kernel.config, cfg.gpu.name): evidence}
    joined = stub(HtsimStepSink(cfg))
    result = joined(record())
    plain = stub(HtsimStepSink(config(tmp_path / "plain")))
    assert result == plain(record())
    measured, = joined.bottleneck_reports[-1].step.kernels
    assert measured == evidence and measured.source_state == "void"
    assert measured.achieved_fraction is not None
    payload = bottleneck_step_to_json(result, joined.bottleneck_reports[-1])
    _, restored = bottleneck_step_from_json(payload)
    assert restored.step.kernels[0].source_state == "void"


def test_late_lifetime_failure_leaves_bottleneck_history_unpublished(monkeypatch):
    from simllm.core import RequestLifetimeRegistry
    runtime = CoarseDeviceRuntime()
    graph = ExecutionGraph("atomic", 0, 0,
                           (operation("fast", 0, 10, "a"), operation("slow", 1, 100, "b")),
                           completion_operation_ids=("fast", "slow"))
    execution = runtime.execute(graph)
    segments = classify_runtime(graph, runtime.last_report, runtime.selected_critical_visits, GPU_ENVELOPES["b100"])
    lifetimes = RequestLifetimeRegistry((0,))

    def reject(*args):
        raise ValueError("rejected lifetime")
    monkeypatch.setattr(lifetimes, "consume_step", reject)
    reducer = CompletionReducer(VirtualClock(), lifetimes=lifetimes, emit_bottleneck_report=True)
    with pytest.raises(ValueError, match="rejected lifetime"):
        reducer.reduce(record(), graph, execution, runtime.last_report, bottleneck_segments=segments)
    assert reducer.bottleneck_reports == []
    assert reducer._bottleneck_history._states == {}
    assert reducer.latest_request_metrics == () and reducer.clock.now_ps == 0


def load_study():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples/bottleneck_report_v1/run_study.py"
    spec = importlib.util.spec_from_file_location("bottleneck_study_test", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    return study


@pytest.mark.parametrize("name", ["m4-w2-200g", "width-all-to-all-w8-400g"])
def test_retained_native_cells_replay_without_backend(tmp_path, name):
    study = load_study()
    fixture = json.loads((study.HERE / "fixtures.json").read_bytes())
    saved = json.loads((study.HERE / "results.json").read_bytes())
    cell = next(c for c in fixture["cells"] if c["case"]["name"] == name)
    result, reconstructed = study.packet_cell(cell["case"], tmp_path, cell)
    expected = next(c for c in saved["packet_cells"] if c["case"]["name"] == name)
    assert result == expected and json.loads(study.canonical(reconstructed)) == cell


def test_tracked_report_text_regenerates_with_lf():
    study = load_study()
    report = json.loads((study.HERE / "results.json").read_bytes())
    assert all(c["held"] for c in study.behavioral_checks(report["packet_cells"], report["coarse_cells"]))
    assert study.exact_oracles(report["packet_cells"], report["coarse_cells"]) == report["exact_oracles"]
    assert study.results_markdown(report).encode() == (study.HERE / "RESULTS.md").read_bytes().replace(b"\r\n", b"\n")
    payload = study.bottleneck_report_to_json(study.bottleneck_report_from_json(report["record_example"]))
    assert payload == report["record_example"]
