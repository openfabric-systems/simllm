"""Deferred local prices preserve service and cannot publish before due time."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig, StepNetworkOutcome
from simllm.compute import ModelDims, RooflineProvider
from simllm.core.clock import VirtualClock
from simllm.core.engine_steps import EngineStepRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.value_snapshot import value_snapshot
from simllm.placement import declared_manifest


def make_sink(path):
    dims = ModelDims(num_layers=1, hidden_size=64, intermediate_size=128,
                     num_heads=4, num_kv_heads=4, head_size=16, vocab_size=256, dtype_bytes=2)
    return HtsimStepSink(HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=(0, 1), dims=dims, workdir=path,
        provider=RooflineProvider(), placement_manifest=declared_manifest(tp=2, nodes=1, gpus_per_node=2),
        collective_fixed_cost_envelope="intra-node-fixed-cost-v1", collective_fixed_cost_arm="lower"))


def record(index=0, release=0):
    return StepRecord(index, release, [ScheduledRequest("request", RequestPhase.DECODE, 1, context_length=16)],
                      num_sampled=1)


def bind_price(sink, runtime, price):
    receipt = None

    def publish(result):
        assert sink.publish_deferred(price, runtime, receipt) is result

    receipt = runtime.submit("engine", price.record, price.result, guard=price.validate, publish=publish,
                             publications=(sink.deferred_publication(price),))
    return receipt


def test_local_price_matches_existing_sink_and_delays_every_projection(tmp_path):
    eager, deferred = make_sink(tmp_path / "eager"), make_sink(tmp_path / "deferred")
    original = record()
    expected = eager(deepcopy(original))
    runtime = EngineStepRuntime(VirtualClock())
    price = deferred.prepare_deferred(original)
    assert price.result == expected
    assert deferred.outcomes == deferred.locality_outcomes == deferred.collective_timing_outcomes == []
    receipt = bind_price(deferred, runtime, price)
    runtime.advance_to(price.result.completed_at_ps - 1)
    assert runtime.complete_due() == () and deferred.outcomes == []
    runtime.advance_to(price.result.completed_at_ps)
    assert runtime.complete_due() == (receipt,)
    assert deferred.outcomes == eager.outcomes
    assert deferred.locality_outcomes == eager.locality_outcomes
    assert deferred.collective_timing_outcomes == eager.collective_timing_outcomes
    assert runtime.results[0][1] == expected


@pytest.mark.parametrize("mutation", ["provider-values", "provider-identity", "resolved-config",
                                    "precision-selection", "prepared-payload", "early-row", "old-row"])
def test_pending_price_checks_values_and_prior_row_contents(tmp_path, mutation):
    sink = make_sink(tmp_path)
    if mutation == "old-row":
        sink(record())
    current = record(index=1 if mutation == "old-row" else 0)
    price = sink.prepare_deferred(current)
    runtime = EngineStepRuntime(VirtualClock())
    bind_price(sink, runtime, price)
    if mutation == "provider-values":
        sink.config.provider.efficiency = 0.8
    elif mutation == "provider-identity":
        sink.config.provider = RooflineProvider()
    elif mutation == "resolved-config":
        sink.config.resolved_collective_latency_profile = replace(
            sink.config.resolved_collective_latency_profile, profile_id="changed")
    elif mutation == "precision-selection":
        sink.config.selected_precision_levels["invented"] = "changed"
    elif mutation == "prepared-payload":
        object.__setattr__(price.simulation, "outcome", replace(price.simulation.outcome, makespan_ps=1))
    elif mutation == "early-row":
        sink.outcomes.append(StepNetworkOutcome(99, 1, None, 1, 0))
    else:
        sink.outcomes[0] = replace(sink.outcomes[0], host_profile="changed")
    runtime.advance_to(price.result.completed_at_ps)
    with pytest.raises((RuntimeError, ValueError), match="changed|replaced|before completion"):
        runtime.complete_due()
    assert runtime.failure and runtime.visits == () and runtime.results == ()


def test_publishing_before_due_cannot_use_a_known_receipt(tmp_path):
    sink, runtime = make_sink(tmp_path), EngineStepRuntime(VirtualClock())
    price = sink.prepare_deferred(record())
    receipt = bind_price(sink, runtime, price)
    with pytest.raises(RuntimeError, match="outside its owned"):
        sink.publish_deferred(price, runtime, receipt)
    assert sink.outcomes == [] and runtime.failure


def test_empty_drain_price_is_consumed_once_by_core_authority(tmp_path):
    sink, runtime = make_sink(tmp_path), EngineStepRuntime(VirtualClock())
    price = sink.prepare_deferred(StepRecord(0, 0, finished_request_ids=["done"]))
    calls, receipt = [], None

    def publish(result):
        calls.append(sink.publish_deferred(price, runtime, receipt))
        sink.publish_deferred(price, runtime, receipt)

    receipt = runtime.submit("engine", price.record, price.result, guard=price.validate, publish=publish,
                             publications=(sink.deferred_publication(price),))
    with pytest.raises(RuntimeError, match="already published"):
        runtime.complete_due()
    assert len(calls) == 1 and runtime.failure
    assert sink.outcomes == [] and runtime.results == ()


@pytest.mark.parametrize("field,value", [("profile", "rnic-cn"), ("collective_fixed_cost_arm", "off"),
                                       ("emit_packet_breakdown", True), ("flow_session", object())])
def test_unsupported_sink_modes_reject_before_pricing(tmp_path, monkeypatch, field, value):
    sink = make_sink(tmp_path)
    setattr(sink.config, field, value)
    calls = []
    monkeypatch.setattr(sink, "_simulate_step", lambda record: calls.append(record))
    with pytest.raises(NotImplementedError, match="isolated declared local"):
        sink.prepare_deferred(record())
    assert not calls and not sink.outcomes


def test_class_source_binding_replacement_invalidates_pending_price(tmp_path, monkeypatch):
    sink = make_sink(tmp_path)
    price = sink.prepare_deferred(record())
    original = HtsimStepSink._publish_result

    def replacement(self, simulation, record=None):
        return original(self, simulation, record)

    monkeypatch.setattr(HtsimStepSink, "_publish_result", replacement)
    with pytest.raises(RuntimeError, match="source binding changed"):
        price.validate()


def test_deferred_config_snapshot_covers_fields_ordinary_equality_ignores(tmp_path):
    sink = make_sink(tmp_path)
    before = deepcopy(sink.config)
    initial = sink._deferred_state()
    sink.config.resolved_collective_evidence_class = "changed"
    # Provider identity is separate; compare config values with one provider.
    before.provider = sink.config.provider
    assert sink.config == before
    assert sink._deferred_state() != initial
    assert asdict(sink.config)["resolved_collective_evidence_class"] == "changed"


@pytest.mark.parametrize("same_object", [False, True])
def test_coherently_rewritten_price_cannot_replace_core_private_snapshot(tmp_path, same_object):
    sink, runtime = make_sink(tmp_path), EngineStepRuntime(VirtualClock())
    price = sink.prepare_deferred(record())
    publication = sink.deferred_publication(price)
    changed = replace(price.simulation, outcome=replace(
        price.simulation.outcome, makespan_ps=price.simulation.outcome.makespan_ps + 1))
    forged = replace(price, simulation=changed, simulation_state=value_snapshot(changed))
    receipt = None

    def publish(result):
        sink.publish_deferred(price if same_object else forged, runtime, receipt)

    receipt = runtime.submit("engine", price.record, price.result, guard=price.validate,
                             publish=publish, publications=(publication,))
    if same_object:
        object.__setattr__(price, "simulation", changed)
        object.__setattr__(price, "simulation_state", value_snapshot(changed))
    forged.validate()
    price.validate()
    runtime.advance_to(price.result.completed_at_ps)
    with pytest.raises((RuntimeError, ValueError), match="prepared publication values changed|payload is not bound"):
        runtime.complete_due()
    assert runtime.failure and not sink.outcomes and not runtime.results


def test_instance_source_shadow_invalidates_pending_price(tmp_path, monkeypatch):
    sink = make_sink(tmp_path)
    price = sink.prepare_deferred(record())
    monkeypatch.setattr(sink, "_publish_result", lambda simulation, record=None: simulation.result)
    with pytest.raises(RuntimeError, match="source binding changed"):
        price.validate()


def test_replaced_binding_helper_cannot_hide_changed_publication(tmp_path, monkeypatch):
    sink, runtime = make_sink(tmp_path), EngineStepRuntime(VirtualClock())
    price = sink.prepare_deferred(record())
    bind_price(sink, runtime, price)
    original = sink._publish

    def changed(simulation, record=None):
        return original(replace(simulation, outcome=replace(
            simulation.outcome, makespan_ps=simulation.outcome.makespan_ps + 1)), record)

    monkeypatch.setattr(sink, "_deferred_bindings", lambda: price.bindings)
    monkeypatch.setattr(sink, "_publish", changed)
    price.validate()
    runtime.advance_to(price.result.completed_at_ps)
    with pytest.raises(ValueError, match="publication values changed"):
        runtime.complete_due()
    assert runtime.failure and sink.outcomes == [] and runtime.results == ()
