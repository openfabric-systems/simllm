"""Native crossbar contract: coupled grants, shared bytes and atomic rejection."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from simllm.backends.htsim_nvlink import NvlinkSwitchArbitration
from simllm.backends.nvswitch import NativeSwitch
from simllm.placement import dgx_peer_fabric


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    if not shutil.which("cmake"):
        pytest.skip("native crossbar tests require CMake and a C++ compiler")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("nvswitch-build")
    subprocess.run(["cmake", "-S", str(root / "simllm/backends/nvswitch"), "-B", str(build),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"],
                   check=True, capture_output=True, text=True)
    name = "simllm_nvswitch.dll" if sys.platform == "win32" else (
        "libsimllm_nvswitch.dylib" if sys.platform == "darwin" else "libsimllm_nvswitch.so")
    return str(next(build.rglob(name)))


def domain():
    return dgx_peer_fabric("a100", node_id="node", ranks=tuple(range(8)),
                           propagation_delay_ps=1000, switch_input_buffer_bytes=65536)


def ports(fabric, source, destination, plane=0, lane=0):
    path = fabric.paths_between(source, destination)[plane * 2 + lane]
    indices = {p.port_id: i for i, p in enumerate(fabric.ports)}
    return indices[path.switch_input_port_id], indices[path.switch_output_port_id]


def head(fabric, token, source, destination, *, plane=0, lane=0, pool=0):
    i, o = ports(fabric, source, destination, plane, lane)
    return token, i, o, pool, 272, 200_000_000_000


def test_parallel_chips_and_shared_output(native_library):
    fabric = domain()
    model = NativeSwitch(native_library, fabric, NvlinkSwitchArbitration.IDENTITY)
    rows = (head(fabric, 0, 0, 1), head(fabric, 1, 2, 1),
            head(fabric, 2, 3, 4), head(fabric, 3, 5, 6, plane=1))
    assert model.plan(0, 25_000_000_000, rows, (65536,)) == ((0, 10880), (2, 10880), (3, 10880))
    assert model.plan(1000, 25_000_000_000, (rows[1],), (65536,)) == ()
    assert model.plan(10880, 25_000_000_000, (rows[1],), (65536,)) == ((1, 21760),)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.plan(0, 1, (), ())


def test_shared_receiver_pool_and_credit_mask(native_library):
    fabric = domain()
    model = NativeSwitch(native_library, fabric, NvlinkSwitchArbitration.IDENTITY)
    rows = (head(fabric, 0, 0, 1, pool=0), head(fabric, 1, 2, 1, plane=1, pool=0),
            head(fabric, 2, 3, 4, pool=1))
    assert model.plan(0, 25_000_000_000, rows, (0, 272)) == ((2, 10880),)
    assert model.plan(0, 25_000_000_000, rows[:2], (272,)) == ((0, 10880),)
    assert model.plan(10880, 25_000_000_000, rows[1:2], (272,)) == ((1, 21760),)


@pytest.mark.parametrize("corruption", ("negative", "cross_chip", "duplicate", "overflow", "rate", "pool"))
def test_bad_call_leaves_no_partial_port_or_cursor_update(native_library, corruption):
    fabric = domain()
    model = NativeSwitch(native_library, fabric, NvlinkSwitchArbitration.ROUND_ROBIN_CANDIDATE)
    good = head(fabric, 0, 0, 1)
    bad = list(head(fabric, 1, 2, 3))
    now, rate = 0, 25_000_000_000
    if corruption == "negative":
        bad[0] = -1
    elif corruption == "cross_chip":
        bad[2] = ports(fabric, 2, 3, 1)[1]
    elif corruption == "duplicate":
        bad[0] = 0
    elif corruption == "overflow":
        now = 2**64 - 1
    elif corruption == "rate":
        rate = 0
    else:
        bad[3] = 2
    with pytest.raises(ValueError):
        model.plan(now, rate, (good, tuple(bad)), (65536,))
    assert model.plan(0, 25_000_000_000, (good,), (65536,)) == ((0, 10880),)


def test_rotating_order_and_time_rewind(native_library):
    fabric = domain()
    model = NativeSwitch(native_library, fabric, NvlinkSwitchArbitration.ROUND_ROBIN_CANDIDATE)
    rows = tuple(head(fabric, i, i + 1, 0) for i in range(3))
    assert [model.plan(t * 10880, 25_000_000_000, rows, (65536,))[0][0]
            for t in range(3)] == [0, 1, 2]
    with pytest.raises(ValueError, match="rewind"):
        model.plan(0, 25_000_000_000, rows, (65536,))


def test_missing_library_and_direct_domain_reject(native_library):
    with pytest.raises(ValueError, match="missing"):
        NativeSwitch(native_library + ".absent", domain(), NvlinkSwitchArbitration.IDENTITY)
    from test_peer_packet_runtime import fabric
    with pytest.raises(ValueError, match="switched"):
        NativeSwitch(native_library, fabric(), NvlinkSwitchArbitration.IDENTITY)
    model = NativeSwitch(native_library, domain(), NvlinkSwitchArbitration.IDENTITY)
    invalid = list(head(domain(), 0, 0, 1))
    invalid[1] = 0  # Physical port zero belongs to a GPU, not a switch.
    with pytest.raises(ValueError, match="physical switch head"):
        model.plan(0, 25_000_000_000, (tuple(invalid),), (65536,))
