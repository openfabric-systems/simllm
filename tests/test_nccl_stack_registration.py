"""The mirrored regMr seam: the registration boundary and its exact off path.

The interim collective-completion contract gates a collective on its
destination memory being registered. This file checks that the gate exists at
the seam where real NCCL puts it, that the seam declares a cost without
spending it, and above all that a communicator which does not ask for the gate
emits exactly the events it emitted before the gate existed.
"""

from __future__ import annotations

import pytest

from simllm.compute import (
    NcclRoute,
    NcclStack,
    NcclStackConfig,
    NcclStackEventKind,
    nccl_stack_events_to_json,
    ncclAllReduce,
    ncclCommInitRank,
    ncclNetRegMr,
    validate_nccl_stack_events,
)
from simllm.core import VirtualClock

CONFIG = NcclStackConfig(channel_count=2, chunk_bytes=1_024, fifo_slots_per_channel=2)
DECLARED_COST_PS = 20_000_000


def _stack() -> NcclStack:
    return NcclStack(clock=VirtualClock(), config=CONFIG)


def _communicator(stack: NcclStack, *, gated: bool):
    return ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="ring-4",
        rank=0,
        require_buffer_registration=gated,
    )


def test_an_ungated_communicator_emits_the_same_events_as_before() -> None:
    default_stack = _stack()
    default_communicator = ncclCommInitRank(
        default_stack,
        nranks=4,
        communicator_id="ring-4",
        rank=0,
    )
    ncclAllReduce(
        default_communicator,
        payload_bytes=16_384,
        operation_id="op",
        route=NcclRoute.INTER_NODE,
    )

    explicit_stack = _stack()
    explicit_communicator = _communicator(explicit_stack, gated=False)
    ncclAllReduce(
        explicit_communicator,
        payload_bytes=16_384,
        operation_id="op",
        route=NcclRoute.INTER_NODE,
    )

    assert nccl_stack_events_to_json(default_stack.events) == (
        nccl_stack_events_to_json(explicit_stack.events)
    )
    assert not any(
        event.function in ("ncclNet.regMr", "simllmChannelBufferRegistered")
        for event in default_stack.events
    )


def test_an_ungated_communicator_refuses_a_buffer_argument() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=False)
    before = stack.events
    with pytest.raises(ValueError, match="require_buffer_registration"):
        ncclAllReduce(
            communicator,
            payload_bytes=16_384,
            operation_id="op",
            route=NcclRoute.INTER_NODE,
            buffer_id="layer-0:tp-attn",
        )
    assert stack.events == before


def test_a_gated_collective_refuses_an_unregistered_buffer() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=True)
    before = stack.events
    with pytest.raises(ValueError, match="not registered on every channel"):
        ncclAllReduce(
            communicator,
            payload_bytes=16_384,
            operation_id="op",
            route=NcclRoute.INTER_NODE,
            buffer_id="layer-0:tp-attn",
        )
    assert stack.events == before


def test_a_gated_collective_requires_a_named_buffer() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=True)
    with pytest.raises(ValueError, match="must name its destination buffer"):
        ncclAllReduce(
            communicator,
            payload_bytes=16_384,
            operation_id="op",
            route=NcclRoute.INTER_NODE,
        )


def test_registration_covers_every_channel_and_declares_its_cost() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=True)
    result = ncclNetRegMr(
        communicator,
        buffer_id="layer-0:tp-attn",
        buffer_bytes=16_384,
        declared_cost_ps=DECLARED_COST_PS,
    )

    assert result.channel_ids == (0, 1)
    assert result.declared_cost_ps == DECLARED_COST_PS
    assert result.total_declared_cost_ps == 2 * DECLARED_COST_PS
    functions = [event.function for event in result.events]
    assert functions == [
        "ncclNet.regMr",
        "simllmChannelBufferRegistered",
        "ncclNet.regMr",
        "simllmChannelBufferRegistered",
    ]
    completions = [
        event
        for event in result.events
        if event.kind is NcclStackEventKind.SIGNAL_STORE
    ]
    assert [event.value for event in completions] == [DECLARED_COST_PS] * 2
    assert [event.channel_id for event in completions] == [0, 1]
    validate_nccl_stack_events(stack.events)


def test_the_seam_declares_the_cost_without_advancing_the_clock() -> None:
    clock = VirtualClock(start_ps=4_096)
    stack = NcclStack(clock=clock, config=CONFIG)
    communicator = _communicator(stack, gated=True)
    ncclNetRegMr(
        communicator,
        buffer_id="layer-0:tp-attn",
        buffer_bytes=16_384,
        declared_cost_ps=DECLARED_COST_PS,
    )
    assert clock.now_ps == 4_096
    assert {event.timestamp_ps for event in stack.events} == {4_096}


def test_a_registered_buffer_lets_the_gated_collective_run() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=True)
    ncclNetRegMr(
        communicator,
        buffer_id="layer-0:tp-attn",
        buffer_bytes=16_384,
        declared_cost_ps=DECLARED_COST_PS,
    )
    result = ncclAllReduce(
        communicator,
        payload_bytes=16_384,
        operation_id="op",
        route=NcclRoute.INTER_NODE,
        buffer_id="layer-0:tp-attn",
    )
    assert result.completion_event.function == "simllmKernelComplete"
    validate_nccl_stack_events(stack.events)


def test_registering_the_same_buffer_twice_is_refused() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=True)
    ncclNetRegMr(communicator, buffer_id="b", buffer_bytes=16_384)
    before = stack.events
    with pytest.raises(ValueError, match="already registered"):
        ncclNetRegMr(communicator, buffer_id="b", buffer_bytes=16_384)
    assert stack.events == before


def test_registration_rejects_malformed_arguments() -> None:
    stack = _stack()
    communicator = _communicator(stack, gated=True)
    with pytest.raises(ValueError):
        ncclNetRegMr(communicator, buffer_id=" ", buffer_bytes=16_384)
    with pytest.raises(ValueError, match="positive integer"):
        ncclNetRegMr(communicator, buffer_id="b", buffer_bytes=0)
    with pytest.raises(ValueError):
        ncclNetRegMr(
            communicator,
            buffer_id="b",
            buffer_bytes=16_384,
            declared_cost_ps=-1,
        )
    with pytest.raises(TypeError, match="ncclCommInitRank"):
        ncclNetRegMr(object(), buffer_id="b", buffer_bytes=1)


def test_the_gate_flag_must_be_a_boolean() -> None:
    stack = _stack()
    with pytest.raises(TypeError, match="require_buffer_registration"):
        ncclCommInitRank(
            stack,
            nranks=4,
            communicator_id="ring-4",
            rank=0,
            require_buffer_registration=1,
        )
