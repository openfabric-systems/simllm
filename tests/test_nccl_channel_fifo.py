"""Independent source fixtures, finite-window relations and live metric joins."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_peer_packet_runtime import engine

from simllm.backends.nccl_resources import NcclGpuProfile, NcclGpuResources
from simllm.backends.nccl_runtime import NcclChannelRuntime, NcclConnection, NcclExecutionConfig
from simllm.compute.gpu_packet_port import GpuPeerPacketSession
from simllm.traffic.nccl_program import (
    LL_CLEAN_MASK,
    NcclRingProgram,
    balanced_channels,
    protocol_stripes,
)

ROOT = Path(__file__).resolve().parents[1]


def gpu(sms=4, **kwargs):
    values = json.loads((ROOT / "examples/nccl_channel_fifo_v1/expectations.json").read_text())["synthetic_gpu"]
    staging = json.loads((ROOT / "examples/nccl_channel_fifo_v1/expectations_shared_staging.json").read_text())
    values["shared_memory_bytes_per_second"] = staging["shared_memory_bytes_per_second"]
    return NcclGpuProfile(available_sms=sms, **{**values, **kwargs})


def runtime(sms=4, width=2, rate=25_000_000_000, **kwargs):
    physical = engine(ranks=width, rate=rate, feed=100_000_000_000)
    session = GpuPeerPacketSession("source", physical.profile, physical.physical)
    return NcclChannelRuntime(session, gpu(sms, **kwargs))


@pytest.mark.parametrize("size,wire,loads", [(4, 2048, 16), (120, 2048, 128),
                                            (1920, 2048, 1920), (1924, 4096, 1936)])
def test_ll128_full_warp_store_and_masked_loads(size, wire, loads):
    stripes = protocol_stripes(size, "LL128", 4)
    assert sum(s.encoded_bytes for s in stripes) == wire
    assert sum(s.local_load_bytes for s in stripes) == loads
    assert sum(s.useful_bytes for s in stripes) == size
    for stripe in stripes:
        assert stripe.load_masks[1] & 0x80808080 == 0
        assert stripe.load_masks[3] & 0x80808080 == 0


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("protocol", ["LL", "LL128", "SIMPLE"])
@pytest.mark.parametrize("size", [4, 12, 16, 20, 116, 120, 124, 1916, 1920, 1924, 131072, 131076])
def test_ring_source_bytes_and_final_output_cover(width, protocol, size):
    program = NcclRingProgram(size, tuple(range(width)), protocol, (size,), 4)
    assert program.useful_network_bytes == 2 * (width - 1) * size
    for rank in range(width):
        primitives = program.primitives(0, rank)
        # Each output byte is written once at each rank, after reduction or copy.
        assert sum(p.useful_bytes for p in primitives if p.output_store) == size
        if protocol == "SIMPLE":
            assert all(p.steps == 2 for p in primitives)
            assert len(primitives) % (2 * (2 * width - 1)) == 0


def test_four_rank_source_chunk_order_and_empty_simple_slice():
    program = NcclRingProgram(256, (0, 1, 2, 3), "LL128", (256,), 4)
    assert [p.chunk_index for p in program.primitives(0, 0)] == [3, 2, 1, 0, 3, 2, 1]
    simple = replace(program, protocol="SIMPLE")
    assert [p.useful_bytes for p in simple.primitives(0, 0)] == [64, 0] * 7
    assert [p.reduce for p in program.primitives(0, 0)] == [False, True, True, True, False, False, False]


@pytest.mark.parametrize("channels", [1, 2, 4, 8, 12, 16, 24, 32])
@pytest.mark.parametrize("sms", [1, 2, 4, 8, 16, 32])
def test_resident_block_waves_have_independent_closed_form(channels, sms):
    physical = engine(ranks=2)
    resources = NcclGpuResources(physical, gpu(sms))
    finishes = []

    def job(block):
        def done():
            finishes.append(physical.now_ps)
            resources.release(block)
        resources.service(block, "fixture", 1000, done)

    for channel in range(channels):
        block = f"c{channel}"
        resources.admit(block, 0, 4, lambda block=block: job(block))
    physical.advance_until(lambda: len(finishes) == channels)
    assert max(finishes) == ((channels + sms - 1) // sms) * 1_000_000
    assert sum(v.units for v in resources.visits) == channels * 1000


@pytest.mark.parametrize("field,value,warps,expected", [
    ("warps_per_sm", 8, 4, 2), ("registers_per_sm", 4096, 4, 1),
    ("shared_bytes_per_sm", 32768, 4, 2), ("blocks_per_sm", 1, 4, 1),
])
def test_each_independent_residency_limit(field, value, warps, expected):
    profile = gpu(1, blocks_per_sm=8)
    assert replace(profile, **{field: value}).residency(warps) == expected


@pytest.mark.parametrize("protocol,steps,capacity", [("LL", 1, 8), ("LL128", 1, 8), ("SIMPLE", 2, 4)])
def test_reserved_capacity_and_remote_head_are_distinct(protocol, steps, capacity):
    connection = NcclConnection(("comm", 0, 0, 1, 0))
    primitive = NcclRingProgram(64, (0, 1), protocol, (64,), 4).primitives(0, 0)[0]
    reservations = [connection.reserve(primitive, protocol, "o") for _ in range(capacity)]
    assert [r.sequence % 8 for r in reservations] == list(range(0, 8, steps))
    with pytest.raises(RuntimeError, match="capacity"):
        connection.reserve(primitive, protocol, "o")
    reservations[0].ready = True
    connection.consume(0)
    assert not connection.can_reserve(steps)
    connection.return_head(steps)
    assert not connection.can_reserve(steps)
    connection.cached_head = connection.returned_head
    assert connection.can_reserve(steps)
    assert connection.reserve(primitive, protocol, "o").sequence == 8


@pytest.mark.parametrize("protocol,warps", [("LL", 4), ("LL128", 4), ("SIMPLE", 5)])
@pytest.mark.parametrize("width", [2, 4])
def test_source_protocol_reaches_output_and_separates_wire_control(protocol, warps, width):
    model = runtime(width=width)
    program = NcclRingProgram(1024, tuple(range(width)), protocol, (512, 512), warps)
    result = model.run("exec", "reduce", program)
    result.validate()
    assert result.completed_at_ps >= max(p.visible_at_ps for p in model.session.packets)
    assert sum(p.payload_bytes for p in model.session.packets) == (
        result.protocol_data_bytes + result.counter_bytes + result.cleanup_bytes)
    # Buffered LL128 does not execute a proxy-only tail or fence.
    if protocol == "LL128":
        assert not any("tail" in v.kind or "fence" in v.kind for v in result.resource_visits)
    if protocol == "SIMPLE":
        assert any(v.kind == "tail_publication" for v in result.resource_visits)
        assert any(v.kind == "fence_and_tail_publication" for v in result.resource_visits)
    model.session.drain()


def test_connection_steps_survive_ll_simple_ll128_calls_and_alignment():
    model = runtime()
    for index, protocol in enumerate(["LL", "SIMPLE", "LL128"]):
        program = NcclRingProgram(64, (0, 1), protocol, (64,), 4)
        result = model.run("exec", str(index), program)
        sequences = [row["sequence"] for row in result.fifo_events if row["kind"] == "reserve"]
        assert min(sequences) == [0, 4, 12][index]
    assert len(model.connections) == 2
    assert {c.producer_step for c in model.connections.values()} == {14}
    model.session.drain()
    assert all(c.consumer_step == c.returned_head == 14 for c in model.connections.values())


def test_ll_flag_wrap_cleanup_is_real_peer_work():
    model = runtime()
    program = NcclRingProgram(64, (0, 1), "LL", (64,), 4)
    for src, dst in [(0, 1), (1, 0)]:
        connection = model.connection(program, 0, src, dst)
        connection.producer_step = connection.consumer_step = LL_CLEAN_MASK
        connection.cached_head = connection.returned_head = LL_CLEAN_MASK
    result = model.run("wrap", "o", program)
    assert result.cleanup_bytes == 4 * (65536 - 64)
    model.session.drain()


def test_invalid_branch_and_duplicate_operation_do_not_admit_new_packets():
    model = runtime()
    with pytest.raises(ValueError, match="buffered"):
        NcclExecutionConfig(gpu(), connection_mode="direct_read")
    program = NcclRingProgram(64, (0, 1), "LL", (64,), 4)
    model.run("exec", "o", program)
    before = model.session.packets
    with pytest.raises(ValueError, match="unique"):
        model.run("exec", "o", program)
    assert model.session.packets == before


def test_callback_cannot_advance_its_own_calendar():
    physical = engine(ranks=2)
    physical.schedule_callback(0, lambda: physical.advance_to(1000))
    with pytest.raises(RuntimeError, match="cannot advance"):
        physical.advance_to(0)


def test_receiver_delay_holds_resident_blocks_and_increases_only_dependent_work():
    results = []
    for delay in [0, 2_000_000]:
        model = runtime(sms=1)
        program = NcclRingProgram(1024, (0, 1), "LL128", balanced_channels(1024, 2), 4)
        results.append(model.run("delay", "o", program, receiver_delay_ps=delay))
    assert results[1].duration_ps > results[0].duration_ps
    assert results[1].protocol_data_bytes == results[0].protocol_data_bytes
    assert sum(v.units for v in results[1].resource_visits if v.kind == "flag_poll") > sum(
        v.units for v in results[0].resource_visits if v.kind == "flag_poll")


def test_program_snapshots_descriptor_sequences_and_rejects_extra_ll_warp():
    ranks, partition = [0, 1], [32, 32]
    program = NcclRingProgram(64, ranks, "LL", partition, 4)
    ranks.reverse()
    partition[:] = [64]
    assert program.ranks == (0, 1)
    assert program.channel_bytes == (32, 32)
    with pytest.raises(ValueError, match="warp"):
        replace(program, warps=17)


@pytest.mark.parametrize("protocol", ["LL", "LL128", "SIMPLE"])
def test_constructor_refreshes_returned_capacity_before_waiting(protocol):
    model = runtime()
    program = NcclRingProgram(64, (0, 1), protocol, (64,), 4)
    # A previous completed invocation exhausted the old cached window.
    for src, dst in [(0, 1), (1, 0)]:
        conn = model.connection(program, 0, src, dst)
        conn.producer_step = conn.consumer_step = conn.returned_head = 8
        conn.cached_head = 0
    result = model.run("constructor", "o", program)
    assert not any(v.kind == "head_poll" for v in result.resource_visits)
    assert all(row["cached_head"] == 8 for row in result.fifo_events
               if row["kind"] == "constructor_head_load")
    model.session.drain()


@pytest.mark.parametrize("protocol", ["LL", "LL128", "SIMPLE"])
def test_empty_tail_ranks_execute_control_and_complete(protocol):
    model = runtime(width=4)
    result = model.run("empty", "o", NcclRingProgram(4, (0, 1, 2, 3), protocol, (4,), 4))
    result.validate()
    assert result.useful_network_bytes == 24
    assert sum(v.units for v in result.resource_visits if v.kind == "output_store") == 16
    model.session.drain()


@pytest.mark.parametrize("useful,shared_store,shared_load", [
    (4, 1920, 4), (12, 1920, 12), (16, 1904, 0), (20, 1904, 4),
    (120, 1808, 8), (1024, 896, 0), (1916, 16, 12), (1920, 0, 0),
])
def test_ll128_store_regs_stages_all_nonoutput_vectors(useful, shared_store, shared_load):
    stripe, = protocol_stripes(useful, "LL128", 4)
    assert stripe.shared_staging_bytes == shared_store
    assert stripe.shared_tail_load_bytes == shared_load


@pytest.mark.parametrize("size", [16, 128, 1920])
@pytest.mark.parametrize("rate", [500_000_000_000, 1_000_000_000_000, 2_000_000_000_000])
@pytest.mark.parametrize("sms", [1, 2])
def test_shared_memory_rate_and_sm_domains_have_independent_oracle(size, rate, sms):
    calendar = engine(ranks=2)
    resources = NcclGpuResources(calendar, gpu(sms, shared_memory_bytes_per_second=rate))
    completions = []
    for index in range(2):
        block = f"b{index}"

        def job(block=block):
            def finish():
                completions.append(calendar.now_ps)
                resources.release(block)
            resources.service(block, "shared", size, finish, shared=True)

        resources.admit(block, 0, 4, job)
    calendar.advance_until(lambda: len(completions) == 2)
    service = (size * 10**12 + rate - 1) // rate
    assert max(completions) == service * (2 if sms == 1 else 1)
    assert sum(v.units for v in resources.visits) == 2 * size
