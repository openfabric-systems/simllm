"""The one-time collective registration cost, its ledger and its off path.

Everything here is component evidence for the mechanism: the ledger's
set-membership behavior is forced by its own construction, which is why the
study registers these as fatal guards rather than as scored relations. What
they defend is that the construction is the one the freeze describes.
"""

from __future__ import annotations

import pytest

from simllm.core import CollectiveWork
from simllm.traffic import (
    DECLARED_CHANNEL_REGISTRATION_COST_PS,
    DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
    FIRST_USE_REASON,
    NCCL_CHANNEL_REGISTRATION_MODEL,
    NEW_BUFFER_REASON,
    NEW_PEER_REASON,
    REBUILD_REASON,
    CollectiveRegistrationCost,
    CollectiveRegistrationIdentity,
    CollectiveRegistrationLedger,
    CollectiveRegistrationModel,
    CollectiveRegistrationProvenance,
    collective_buffer_id,
    collective_communicator_id,
    resolve_collective_registration,
)

COST_PS = DECLARED_CHANNEL_REGISTRATION_COST_PS


def _work(ranks: tuple[int, ...] = (0, 1)) -> CollectiveWork:
    return CollectiveWork(
        collective="all-reduce",
        ranks=ranks,
        payload_bytes=1024,
        algorithm_hint="ring",
    )


def _ledger(channels: int = 1) -> CollectiveRegistrationLedger:
    model = CollectiveRegistrationModel(
        model_id="test-registration",
        cost=DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
        channels_per_communicator=channels,
    )
    return CollectiveRegistrationLedger(model=model)


def test_the_declared_cost_is_twenty_microseconds_and_says_so() -> None:
    cost = DECLARED_NCCL_CHANNEL_REGISTRATION_COST
    assert cost.registration_cost_ps == 20_000_000
    assert cost.evidence_class == "declared"
    assert cost.declared_cost_ps() == 20_000_000


def test_asking_a_declared_cost_for_a_calibrated_value_fails_closed() -> None:
    with pytest.raises(ValueError, match="not calibrated"):
        DECLARED_NCCL_CHANNEL_REGISTRATION_COST.calibrated_cost_ps()


def test_a_calibrated_cost_must_name_its_measurement() -> None:
    with pytest.raises(ValueError, match="must name the measurement"):
        CollectiveRegistrationProvenance(
            evidence_class="calibrated",
            source="a capture",
            locator="a row",
            basis="fitted",
        )


def test_a_declared_cost_cannot_carry_a_measurement() -> None:
    with pytest.raises(ValueError, match="cannot name a measurement"):
        CollectiveRegistrationProvenance(
            evidence_class="declared",
            source="a guess",
            locator="nowhere",
            basis="none",
            measurement="a capture",
        )


def test_a_calibrated_cost_returns_its_value() -> None:
    cost = CollectiveRegistrationCost(
        cost_id="hypothetical-calibrated",
        registration_cost_ps=1_234,
        provenance=CollectiveRegistrationProvenance(
            evidence_class="calibrated",
            source="a hypothetical capture",
            locator="a hypothetical row",
            basis="fitted to the capture that prices this registration",
            measurement="the hypothetical capture",
        ),
    )
    assert cost.calibrated_cost_ps() == 1_234


def test_an_unknown_model_selector_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown collective registration model"):
        resolve_collective_registration("no-such-model")
    assert resolve_collective_registration(None) is None
    assert (
        resolve_collective_registration("nccl-channel-registration-v1")
        is NCCL_CHANNEL_REGISTRATION_MODEL
    )


def test_a_disabled_ledger_charges_nothing_and_records_nothing() -> None:
    ledger = CollectiveRegistrationLedger()
    assert not ledger.enabled
    assert ledger.charge_collective(_work(), "step-0:layer-0:tp-attn") == 0
    assert ledger.charge_collective(_work(), "step-1:layer-0:tp-attn") == 0
    assert ledger.events == ()
    assert ledger.charged_ps == 0
    assert ledger.identities_for(communicator_id="c", buffer_id="b") == ()


def test_the_first_collective_pays_once_and_the_second_pays_nothing() -> None:
    ledger = _ledger()
    first = ledger.charge_collective(_work(), "step-0:layer-0:tp-attn")
    second = ledger.charge_collective(_work(), "step-1:layer-0:tp-attn")
    third = ledger.charge_collective(_work(), "step-2:layer-0:tp-attn")
    assert (first, second, third) == (COST_PS, 0, 0)
    assert ledger.charged_ps == COST_PS
    assert [event.reason for event in ledger.events] == [FIRST_USE_REASON]


def test_two_channels_pay_twice() -> None:
    ledger = _ledger(channels=2)
    assert ledger.charge_collective(_work(), "step-0:layer-0:tp-attn") == 2 * COST_PS
    assert ledger.charge_collective(_work(), "step-1:layer-0:tp-attn") == 0
    assert [identity.channel_id for identity in ledger.registered_identities] == [0, 1]


def test_a_new_buffer_re_registers_only_itself() -> None:
    ledger = _ledger()
    ledger.charge_collective(_work(), "step-0:layer-0:tp-attn")
    charged = ledger.charge_collective(_work(), "step-0:layer-0:tp-mlp")
    assert charged == COST_PS
    assert ledger.charge_collective(_work(), "step-1:layer-0:tp-attn") == 0
    assert [event.reason for event in ledger.events] == [
        FIRST_USE_REASON,
        NEW_BUFFER_REASON,
    ]


def test_a_new_peer_set_re_registers_every_buffer() -> None:
    ledger = _ledger()
    ledger.charge_collective(_work((0, 1)), "step-0:layer-0:tp-attn")
    charged = ledger.charge_collective(_work((0, 1, 2, 3)), "step-0:layer-0:tp-attn")
    assert charged == COST_PS
    assert [event.reason for event in ledger.events] == [
        FIRST_USE_REASON,
        NEW_PEER_REASON,
    ]


def test_a_communicator_rebuild_re_registers_every_channel_and_buffer() -> None:
    ledger = _ledger(channels=2)
    work = _work()
    ledger.charge_collective(work, "step-0:layer-0:tp-attn")
    ledger.charge_collective(work, "step-0:layer-0:tp-mlp")
    assert ledger.charged_ps == 4 * COST_PS

    generation = ledger.rebuild_communicator(collective_communicator_id(work))
    assert generation == 1
    assert ledger.charge_collective(work, "step-1:layer-0:tp-attn") == 2 * COST_PS
    assert ledger.charge_collective(work, "step-1:layer-0:tp-mlp") == 2 * COST_PS
    assert ledger.charge_collective(work, "step-2:layer-0:tp-attn") == 0
    assert ledger.charged_ps == 8 * COST_PS
    assert {event.reason for event in ledger.events[4:]} == {REBUILD_REASON}


def test_rebuild_all_invalidates_every_communicator_the_ledger_charged() -> None:
    ledger = _ledger()
    ledger.charge_collective(_work((0, 1)), "step-0:layer-0:tp-attn")
    ledger.charge_collective(_work((0, 1, 2, 3)), "step-0:layer-0:tp-attn")
    ledger.rebuild_all()
    assert ledger.charge_collective(_work((0, 1)), "step-1:layer-0:tp-attn") == COST_PS
    assert (
        ledger.charge_collective(_work((0, 1, 2, 3)), "step-1:layer-0:tp-attn")
        == COST_PS
    )


def test_the_communicator_identity_is_the_kind_and_the_rank_set() -> None:
    assert collective_communicator_id(_work((0, 1))) == "all-reduce:[0-1]"
    assert collective_communicator_id(_work((0, 1, 2))) == "all-reduce:[0-1-2]"


def test_the_buffer_identity_is_the_site_not_the_occurrence() -> None:
    assert collective_buffer_id("step-0:layer-3:tp-mlp") == "layer-3:tp-mlp"
    assert collective_buffer_id("step-77:layer-3:tp-mlp") == "layer-3:tp-mlp"


def test_an_operation_identity_without_a_step_component_fails_closed() -> None:
    with pytest.raises(ValueError, match="carries no 'step-<n>:' component"):
        collective_buffer_id("layer-3:tp-mlp")


def test_registration_identities_reject_malformed_fields() -> None:
    with pytest.raises(ValueError):
        CollectiveRegistrationIdentity("", 0, 0, "b")
    with pytest.raises(ValueError):
        CollectiveRegistrationIdentity("c", -1, 0, "b")
    with pytest.raises(ValueError):
        CollectiveRegistrationIdentity("c", 0, -1, "b")
    with pytest.raises(ValueError):
        CollectiveRegistrationIdentity("c", 0, 0, " ")


def test_the_shipped_model_registers_one_channel_by_default() -> None:
    model = NCCL_CHANNEL_REGISTRATION_MODEL
    assert model.channels_per_communicator == 1
    assert model.channel_ids == (0,)
    assert model.first_use_cost_ps() == COST_PS
    assert model.cost.evidence_class == "declared"
