"""Product wiring and native/Python conformance through the retained calendar."""

from collections import Counter
from dataclasses import asdict, replace
from itertools import permutations

import pytest
from test_nvswitch_native import native_library  # noqa: F401

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkSwitchArbitration,
    NvlinkTransfer,
)
from simllm.backends.nvlink_runtime import NvlinkCausalEngine, NvlinkPhysicalBinding
from simllm.placement import PeerFabric, dgx_peer_fabric


def make_engine(generation, *, library=None, capacity=65536, rx_capacity=65536,
                rate=25_000_000_000, arbitration=NvlinkSwitchArbitration.IDENTITY, processing=0):
    from examples.local_peer_packet_runtime_v1.run_study import profile_for
    fabric = dgx_peer_fabric(generation, node_id="node-0", ranks=tuple(range(8)),
                            propagation_delay_ps=1000, switch_input_buffer_bytes=capacity)
    fabric = replace(fabric, links=tuple(replace(link, link_rate_bps=rate * 8) for link in fabric.links))
    endpoint_rate = (300 if generation == "a100" else 450) * 10**9
    profile = profile_for(True, rate=rate, feed=endpoint_rate, rx_rate=endpoint_rate,
                          capacity=capacity, receiver_capacity=rx_capacity)
    return NvlinkCausalEngine(profile, NvlinkAlignedOptions(switch_arbitration=arbitration),
                             physical=NvlinkPhysicalBinding(fabric, processing),
                             native_switch_library=library)


@pytest.mark.parametrize("generation,links,chips,paths", (("a100", 96, 6, 12), ("h100", 144, 4, 18)))
def test_product_attachment_counts_paths_and_roundtrip(generation, links, chips, paths):
    fabric = dgx_peer_fabric(generation, node_id="board", ranks=tuple(range(16, 24)),
                            propagation_delay_ps=1000, switch_input_buffer_bytes=65536)
    assert len(fabric.links) == links
    assert len({p.switch_id for p in fabric.ports if p.switch_id is not None}) == chips
    assert Counter(p.gpu_rank for p in fabric.ports if p.gpu_rank is not None) == dict.fromkeys(range(16, 24), paths)
    assert len(fabric.routes) == 56
    for a, b in permutations(range(16, 24), 2):
        routes = fabric.paths_between(a, b)
        assert len(routes) == paths
        assert len({path.input_resource for path in routes}) == paths
        assert len({path.output_resource for path in routes}) == paths
        assert all(path.switch_id and path.output_link for path in routes)
    assert PeerFabric.from_dict(asdict(fabric)) == fabric
    assert all(link.link_rate_bps == 200_000_000_000 for link in fabric.links)


@pytest.mark.parametrize("ranks", ((0, 1), tuple(range(7)), (0,) * 8, (True, *range(1, 8))))
def test_preset_does_not_turn_arbitrary_rank_counts_into_a_dgx(ranks):
    with pytest.raises(ValueError, match="eight distinct"):
        dgx_peer_fabric("a100", node_id="board", ranks=ranks,
                        propagation_delay_ps=0, switch_input_buffer_bytes=272)


@pytest.mark.parametrize("generation", ("a100", "h100"))
@pytest.mark.parametrize("policy", tuple(NvlinkSwitchArbitration))
@pytest.mark.parametrize("capacity", (272, 65536))
def test_native_matches_every_packet_and_buffer_boundary(native_library, generation, policy, capacity):  # noqa: F811
    transfers = tuple(NvlinkTransfer(extent_id=f"flow-{i}", source=i, destination=0,
                                    payload_bytes=4096, topology_endpoint_count=8) for i in range(1, 8))
    outputs = []
    for library in (None, native_library):
        engine = make_engine(generation, library=library, capacity=capacity,
                             rx_capacity=capacity, arbitration=policy)
        engine.admit(transfers, include_switch=True)
        first = engine.advance_until_visible(tuple(t.extent_id for t in transfers))
        next_transfer = NvlinkTransfer(extent_id="next", source=0, destination=7, payload_bytes=256,
                                       released_at_ps=first, topology_endpoint_count=8)
        engine.admit((next_transfer,), include_switch=True)
        result = engine.drain()
        outputs.append((result, engine.physical_paths, engine.buffer_claims, engine.buffer_ownership))
        if library:
            assert engine._switch_inputs == engine._switch_outputs == {}
    assert outputs[0] == outputs[1]


def test_absent_native_selection_does_not_import_native_binding(monkeypatch):
    import builtins
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if "nvswitch" in name:
            raise AssertionError("disabled native binding imported")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    engine = make_engine("a100")
    engine.admit((NvlinkTransfer(extent_id="one", source=0, destination=1,
                                 payload_bytes=256, topology_endpoint_count=8),), include_switch=True)
    assert engine.drain().packets[0].visible_at_ps > 0
