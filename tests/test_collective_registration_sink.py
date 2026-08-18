"""The registration charge on the live chain, and its byte-identical off path.

These cases drive the real `HtsimStepSink` with a stubbed fabric backend, which
is the same seam the accepted attribution tests use. What they check is the
part the ledger's own unit tests cannot: that a charge decided once per
semantic collective survives the artifact split, the composition, the
conservation guards and the per-request reducer, and that a sink which does not
select a registration model produces exactly the run it produced before the
model existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement
from simllm.traffic import (
    DECLARED_CHANNEL_REGISTRATION_COST_PS,
    DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
    CollectiveRegistrationModel,
)

DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
)

COST_PS = DECLARED_CHANNEL_REGISTRATION_COST_PS
#: two layers, two tensor-parallel sites each
IDENTITIES = 4
STUB_FABRIC_PS = 1_000_000


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="measured")


def _stub_backend(monkeypatch) -> None:
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def run(config):
        text = config.goal_bin.read_text()
        return SimpleNamespace(
            job_completion_time_ps=lambda: STUB_FABRIC_PS,
            flows=[object()] * text.count(": send "),
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", run)


def _manifest(hosts: list[str]) -> PlacementManifest:
    counts: dict[str, int] = {}
    ranks = []
    for global_rank, host in enumerate(hosts):
        local_rank = counts.get(host, 0)
        counts[host] = local_rank + 1
        ranks.append(
            RankPlacement(global_rank=global_rank, hostname=host, local_rank=local_rank)
        )
    return PlacementManifest(ranks=ranks)


def _record(step_index: int, virtual_time_ps: int) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[ScheduledRequest("r0", RequestPhase.DECODE, 3, context_length=8)],
        num_sampled=1,
    )


def _sink(tmp_path, hosts: list[str], label: str, **kwargs) -> HtsimStepSink:
    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=tuple(range(len(hosts))),
            dims=DIMS,
            workdir=tmp_path / label,
            placement_manifest=_manifest(hosts),
            provider=FixedProvider(),
            **kwargs,
        )
    )


def _replay(sink: HtsimStepSink, steps: int = 3, rebuild_before: int | None = None):
    reducer = HtsimRequestMetricReducer({"r0": 0})
    virtual_time_ps = 0
    for step_index in range(steps):
        if rebuild_before is not None and step_index == rebuild_before:
            sink.rebuild_collective_communicators()
        record = _record(step_index, virtual_time_ps)
        result = sink(record)
        assert result is not None
        reducer.consume(record, result, sink.locality_outcomes[step_index])
        virtual_time_ps = result.completed_at_ps
    return reducer.totals()


def _goal_digests(sink: HtsimStepSink) -> list[bytes]:
    return sorted(
        path.read_bytes() for path in sorted(sink.config.workdir.rglob("*.goal"))
    )


def test_the_default_construction_and_an_explicit_none_agree_exactly(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    default = _sink(tmp_path, ["a", "a"], "default")
    explicit = _sink(tmp_path, ["a", "a"], "explicit", collective_registration=None)

    default_totals = _replay(default)
    explicit_totals = _replay(explicit)

    assert default.locality_outcomes == explicit.locality_outcomes
    assert [outcome.makespan_ps for outcome in default.outcomes] == [
        outcome.makespan_ps for outcome in explicit.outcomes
    ]
    assert default_totals == explicit_totals
    assert default.collective_registration_outcomes == []
    assert explicit.collective_registration_outcomes == []
    assert all(
        outcome.registration_phase_cost_ps == ()
        for outcome in default.locality_outcomes
    )
    assert not default.collective_registration_ledger.enabled


def test_the_first_step_pays_once_per_identity_and_later_steps_pay_nothing(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    off = _sink(tmp_path, ["a", "a"], "off")
    on = _sink(
        tmp_path,
        ["a", "a"],
        "on",
        collective_registration="nccl-channel-registration-v1",
    )

    _replay(off)
    _replay(on)

    off_makespans = [outcome.makespan_ps for outcome in off.outcomes]
    on_makespans = [outcome.makespan_ps for outcome in on.outcomes]
    assert on_makespans[0] - off_makespans[0] == IDENTITIES * COST_PS
    assert on_makespans[1:] == off_makespans[1:]

    charged = [
        outcome.charged_ps for outcome in on.collective_registration_outcomes
    ]
    assert charged == [IDENTITIES * COST_PS, 0, 0]
    assert on.collective_registration_ledger.charged_ps == IDENTITIES * COST_PS


def test_registration_changes_no_emitted_byte(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    off = _sink(tmp_path, ["a", "a", "b", "b"], "bytes-off")
    on = _sink(
        tmp_path,
        ["a", "a", "b", "b"],
        "bytes-on",
        collective_registration="nccl-channel-registration-v1",
    )
    _replay(off, steps=2)
    _replay(on, steps=2)

    assert _goal_digests(off) == _goal_digests(on)
    assert [outcome.num_flows for outcome in off.outcomes] == [
        outcome.num_flows for outcome in on.outcomes
    ]
    assert [
        outcome.total_directed_bytes for outcome in off.locality_outcomes
    ] == [outcome.total_directed_bytes for outcome in on.locality_outcomes]


def test_a_phase_split_collective_is_charged_exactly_once(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    on = _sink(
        tmp_path,
        ["a", "a", "b", "b"],
        "phases",
        collective_registration="nccl-channel-registration-v1",
    )
    _replay(on, steps=2)

    first = on.collective_registration_outcomes[0]
    locality = on.locality_outcomes[0]
    #: the mixed placement splits every collective into more artifacts than
    #: there are collectives, which is the shape a per-artifact charge would
    #: get wrong
    assert locality.artifact_count > IDENTITIES
    assert len(locality.registration_phase_cost_ps) == locality.artifact_count
    assert sum(locality.registration_phase_cost_ps) == IDENTITIES * COST_PS
    assert sum(
        1 for value in locality.registration_phase_cost_ps if value
    ) == IDENTITIES
    assert first.charged_ps == IDENTITIES * COST_PS
    assert on.collective_registration_outcomes[1].charged_ps == 0


def test_two_channels_pay_twice_on_the_live_chain(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    off = _sink(tmp_path, ["a", "a"], "one-off")
    two = _sink(
        tmp_path,
        ["a", "a"],
        "two-channel",
        collective_registration=CollectiveRegistrationModel(
            model_id="two-channel-test",
            cost=DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
            channels_per_communicator=2,
        ),
    )
    _replay(off, steps=2)
    _replay(two, steps=2)

    delta = two.outcomes[0].makespan_ps - off.outcomes[0].makespan_ps
    assert delta == 2 * IDENTITIES * COST_PS
    assert two.outcomes[1].makespan_ps == off.outcomes[1].makespan_ps


def test_a_communicator_rebuild_re_pays_on_the_next_step(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    off = _sink(tmp_path, ["a", "a"], "rebuild-off")
    plain = _sink(
        tmp_path,
        ["a", "a"],
        "rebuild-plain",
        collective_registration="nccl-channel-registration-v1",
    )
    rebuilt = _sink(
        tmp_path,
        ["a", "a"],
        "rebuild-on",
        collective_registration="nccl-channel-registration-v1",
    )
    _replay(off)
    _replay(plain)
    _replay(rebuilt, rebuild_before=2)

    assert rebuilt.outcomes[0].makespan_ps == plain.outcomes[0].makespan_ps
    assert rebuilt.outcomes[1].makespan_ps == plain.outcomes[1].makespan_ps
    assert (
        rebuilt.outcomes[2].makespan_ps - off.outcomes[2].makespan_ps
        == IDENTITIES * COST_PS
    )
    assert [
        outcome.charged_ps for outcome in rebuilt.collective_registration_outcomes
    ] == [IDENTITIES * COST_PS, 0, IDENTITIES * COST_PS]


def test_the_attribution_names_the_registration_and_conserves_the_step(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    on = _sink(
        tmp_path,
        ["a", "a", "b", "b"],
        "attribution",
        collective_registration="nccl-channel-registration-v1",
    )
    _replay(on, steps=2)

    for index, locality in enumerate(on.locality_outcomes):
        step = attribute_step_detail(_step_result(on, index), locality)
        charged = on.collective_registration_outcomes[index].charged_ps
        assert step.media.collective_registration_ps == charged
        assert step.media.total_ps == on.outcomes[index].makespan_ps
        assert step.media.collective_ps == step.attribution.collective_ps


def _step_result(sink: HtsimStepSink, index: int):
    from simllm.core import StepResult

    outcome = sink.outcomes[index]
    return StepResult(
        step_index=outcome.step_index,
        step_latency_ps=outcome.makespan_ps,
        completed_at_ps=outcome.makespan_ps,
    )


def test_ttft_moves_by_exactly_the_registered_amount(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    off = _sink(tmp_path, ["a", "a"], "ttft-off")
    on = _sink(
        tmp_path,
        ["a", "a"],
        "ttft-on",
        collective_registration="nccl-channel-registration-v1",
    )
    (off_row,) = _replay(off)
    (on_row,) = _replay(on)

    assert on_row.ttft_ps - off_row.ttft_ps == IDENTITIES * COST_PS
    assert on_row.ttft_media.collective_registration_ps == IDENTITIES * COST_PS
    assert on_row.decode_media.collective_registration_ps == 0
    assert on_row.decode_attribution.total_ps == off_row.decode_attribution.total_ps
    assert on_row.ttft_media.total_ps == on_row.ttft_ps


def test_an_unknown_registration_selector_fails_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="unknown collective registration model"):
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=DIMS,
            workdir=tmp_path / "bad",
            collective_registration="no-such-model",
        )


def test_the_registration_outcome_publishes_its_declared_identity(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    on = _sink(
        tmp_path,
        ["a", "a"],
        "identity",
        collective_registration="nccl-channel-registration-v1",
    )
    _replay(on, steps=1)

    (outcome,) = on.collective_registration_outcomes
    assert outcome.model_id == "nccl-channel-registration-v1"
    assert outcome.cost_id == "nccl-channel-registration-declared-v1"
    assert outcome.evidence_class == "declared"
    assert outcome.registration_cost_ps == COST_PS
    assert outcome.channels_per_communicator == 1
    assert len(outcome.events) == IDENTITIES
    assert {event.step_index for event in outcome.events} == {0}
    assert {
        event.identity.communicator_id for event in outcome.events
    } == {"all-reduce:[0-1]"}
    assert sorted(event.identity.buffer_id for event in outcome.events) == [
        "layer-0:tp-attention",
        "layer-0:tp-mlp",
        "layer-1:tp-attention",
        "layer-1:tp-mlp",
    ]
