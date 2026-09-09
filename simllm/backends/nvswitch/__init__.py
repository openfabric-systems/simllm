"""Optional native switch grant/service kernel, loaded only by explicit selection.

The surrounding causal engine owns packet queues, byte capacity, credits and
calendar events. Native state exclusively owns crossbar port occupancy and its
arbitration cursor. No shadow Python switch-port clock advances when selected.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import weakref
from pathlib import Path

from simllm.backends.htsim_nvlink import NvlinkSwitchArbitration


class Head(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        "token", "input_port", "output_port", "pool", "wire_bytes", "link_rate_bps",
    )]


class Grant(ct.Structure):
    _fields_ = [("token", ct.c_uint64), ("finished_at_ps", ct.c_uint64)]


def _u64(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("native switch inputs must be unsigned 64-bit integers")
    return value


class NativeSwitch:
    """One context per physical domain; no compilation or network access at load."""

    def __init__(self, library: str, fabric, arbitration: NvlinkSwitchArbitration):
        path = Path(library)
        if not path.is_file():
            raise ValueError("selected NVSwitch library is missing; build it with CMake")
        if not fabric.switched:
            raise ValueError("native NVSwitch selection requires a switched peer domain")
        if arbitration not in (NvlinkSwitchArbitration.IDENTITY,
                               NvlinkSwitchArbitration.ROUND_ROBIN_CANDIDATE):
            raise ValueError("unsupported native switch arbitration")
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.arbitration = arbitration
        self._lib = ct.CDLL(str(path.resolve()))
        lib = self._lib
        lib.simllm_nvs_abi_version.restype = ct.c_uint32
        if lib.simllm_nvs_abi_version() != 1:
            raise ValueError("unsupported NVSwitch C ABI")
        lib.simllm_nvs_create.argtypes = [ct.POINTER(ct.c_int32), ct.c_size_t, ct.c_uint32]
        lib.simllm_nvs_create.restype = ct.c_void_p
        lib.simllm_nvs_destroy.argtypes = [ct.c_void_p]
        lib.simllm_nvs_destroy.restype = None
        lib.simllm_nvs_error.argtypes = [ct.c_void_p]
        lib.simllm_nvs_error.restype = ct.c_char_p
        lib.simllm_nvs_plan.argtypes = [
            ct.c_void_p, ct.c_uint64, ct.c_uint64, ct.POINTER(Head), ct.c_size_t,
            ct.POINTER(ct.c_uint64), ct.c_size_t, ct.POINTER(Grant), ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ]
        lib.simllm_nvs_plan.restype = ct.c_int
        chips = {name: i for i, name in enumerate(sorted({p.switch_id for p in fabric.ports
                                                        if p.switch_id is not None}))}
        port_chips = (ct.c_int32 * len(fabric.ports))(
            *(chips[p.switch_id] if p.switch_id is not None else -1 for p in fabric.ports)
        )
        self._context = lib.simllm_nvs_create(
            port_chips, len(port_chips), arbitration is NvlinkSwitchArbitration.ROUND_ROBIN_CANDIDATE,
        )
        if not self._context:
            raise ValueError("cannot create the selected native NVSwitch context")
        self._finalize = weakref.finalize(self, lib.simllm_nvs_destroy, self._context)

    def plan(self, now_ps, crossbar_rate, heads, capacities):
        """Commit grants from numeric read-only queue and receiver snapshots."""
        if not self._finalize.alive:
            raise RuntimeError("native switch context is closed")
        now_ps, crossbar_rate = _u64(now_ps), _u64(crossbar_rate)
        for row in heads:
            if len(row) != 6:
                raise ValueError("native switch head requires six fields")
            for value in row:
                _u64(value)
        for value in capacities:
            _u64(value)
        raw = (Head * len(heads))(*(Head(*row) for row in heads))
        pools = (ct.c_uint64 * len(capacities))(*capacities)
        output = (Grant * len(heads))()
        count = ct.c_size_t()
        status = self._lib.simllm_nvs_plan(
            self._context, now_ps, crossbar_rate, raw, len(raw), pools, len(pools),
            output, len(output), ct.byref(count),
        )
        if status:
            raise ValueError(self._lib.simllm_nvs_error(self._context).decode())
        return tuple((output[i].token, output[i].finished_at_ps) for i in range(count.value))

    def close(self):
        self._finalize()
