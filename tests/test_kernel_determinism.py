"""Locks for the kernel-time determinism contract.

The maintainer ruled on 2026-08-18 that a compute kernel's service time is a
deterministic constant with no tail: a pure function of kernel family, phase,
token and shape inputs, and the architecture profile, identical across ranks and
across GPU runners for the same inputs, with memory-bound kernels pinned to the
HBM bound. This file is what makes that statement fail loudly when it stops
being true.

Four things are locked, matching the four clauses of the contract stated in
`docs/modules/compute.md`.

Determinism: repeated estimates, and estimates from freshly constructed provider
instances, serialize to one byte string. The negative control in
`test_the_determinism_lock_sees_a_changed_efficiency` is what makes that mean
something.

Rank and runner independence: no pricing entry point accepts a caller identity,
no module under `simllm/compute` imports a random-number source, a wall clock or
a process environment reader, and the vLLM and SGLang geometry readers agree on
every value the cost model reads, so both adapters price one step identically.

Phase and token keying: prefill and decode, and a changed token count, move the
constant to the exact values the freeze predicted.

The memory-bound pin: the roofline path returns exactly its HBM-limited time and
reports the memory roof, and the mechanistic `SmSchedulerModel` path returns
exactly its HBM cursor occupancy plus the fixed return latency, unchanged by SM
count.

The fifth block covers the port taxonomy: xGMI and UALink are nameable and fail
closed at configuration time, naming COMP-35.

Fixtures come from `examples/kernel_determinism_v1/run_study.py` so the study
and these locks cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from simllm.compute import (
    GpuPortCapability,
    GpuPortConfig,
    GpuPortDirection,
    GpuPortProtocol,
    GpuPortRole,
    HostInitiationModel,
    ProfileTableProvider,
    RooflineProvider,
    step_kernel,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = ROOT / "examples" / "kernel_determinism_v1" / "run_study.py"
SPEC = importlib.util.spec_from_file_location("kernel_determinism_v1_study", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
# Register before executing: the study defines dataclasses, and dataclasses
# resolves annotations through ``sys.modules`` while the class body runs.
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)

CELL_IDS = [row_id for row_id, _, _, _ in study.ROOFLINE_CELLS]
CELLS = {row_id: (factory, sampled) for row_id, _, factory, sampled in study.ROOFLINE_CELLS}


def _estimate(cell_id: str, provider: RooflineProvider | None = None):
    factory, sampled = CELLS[cell_id]
    return study._roofline_estimate(provider or study.roofline(), factory(), sampled)


# ---------------------------------------------------------------------------
# (a) Determinism, and the control that gives it teeth.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell_id", CELL_IDS)
def test_repeated_roofline_estimates_are_one_byte_string(cell_id):
    provider = study.roofline()
    factory, sampled = CELLS[cell_id]
    record = factory()

    streams = {
        study.canonical(study._roofline_estimate(provider, record, sampled))
        for _ in range(study.REPEATS)
    }

    assert len(streams) == 1


@pytest.mark.parametrize("cell_id", CELL_IDS)
def test_a_fresh_roofline_provider_gives_the_same_byte_string(cell_id):
    assert study.canonical(_estimate(cell_id)) == study.canonical(_estimate(cell_id))


@pytest.mark.parametrize(
    ("bandwidth", "sm_count"), [(64, 1), (32, 1), (64, 2)]
)
def test_the_mechanistic_replay_is_deterministic_across_instances(bandwidth, sm_count):
    launch = study.mechanistic_launch()

    streams = {
        study.canonical(
            study.mechanistic_model(
                hbm_bandwidth=bandwidth, sm_count=sm_count
            ).estimate(launch)
        )
        for _ in range(8)
    }

    assert len(streams) == 1


def test_a_profile_table_lookup_is_deterministic_across_instances():
    """The measured-table provider is a lookup, so it must not drift either."""

    kernel = step_kernel(study.model_dims(), study.decode_record(2), 2)
    key = (kernel.name, kernel.config, study.gpu_spec().name)
    table = {key: 4_242_000}

    first = ProfileTableProvider(dict(table)).estimate(kernel, study.gpu_spec())
    second = ProfileTableProvider(dict(table)).estimate(kernel, study.gpu_spec())

    assert study.canonical(first) == study.canonical(second)
    assert first.duration_ps == 4_242_000
    assert first.bound == "measured"


def test_the_determinism_lock_sees_a_changed_efficiency():
    """The negative control for every byte-equality assertion above."""

    baseline = study.canonical(_estimate("A1"))
    mutated = study.canonical(_estimate("A1", study.roofline(efficiency=0.25)))

    assert baseline != mutated


def test_the_determinism_lock_sees_a_changed_cursor_bandwidth():
    launch = study.mechanistic_launch()

    fast = study.mechanistic_model(hbm_bandwidth=64, sm_count=1).estimate(launch)
    slow = study.mechanistic_model(hbm_bandwidth=32, sm_count=1).estimate(launch)

    assert study.canonical(fast) != study.canonical(slow)


# ---------------------------------------------------------------------------
# (b) Rank and runner independence.
# ---------------------------------------------------------------------------


def test_no_compute_module_imports_a_nondeterminism_source():
    """A duration may not depend on an RNG, a wall clock or the environment."""

    assert study.nondeterminism_audit() == {}


def test_the_nondeterminism_audit_can_see_an_offender(tmp_path):
    """The control: the audit reads imports, it does not return an empty dict."""

    offender = tmp_path / "offender.py"
    offender.write_text(
        "import random\nimport numpy.random\nfrom time import monotonic\n",
        encoding="utf-8",
    )
    clean = tmp_path / "clean.py"
    clean.write_text("import math\nfrom fractions import Fraction\n", encoding="utf-8")

    assert study.nondeterminism_audit([offender]) == {
        "offender.py": ["numpy.random", "random", "time"]
    }
    assert study.nondeterminism_audit([clean]) == {}


def test_the_audit_covers_every_compute_module():
    """An audit that looked at nothing would also report no offender."""

    audited = {path.name for path in study.compute_module_paths()}

    assert {"provider.py", "transformer.py", "gpu_model.py", "host.py"} <= audited
    assert len(audited) >= 8


def test_no_pricing_entry_point_accepts_a_caller_identity():
    """Nothing that turns work into a duration may key on who asked."""

    assert study.identity_parameter_audit() == {}


def test_every_compute_provider_prices_on_kernel_and_gpu_alone():
    import inspect

    from simllm.compute import ComputeProvider

    for subclass in ComputeProvider.__subclasses__():
        parameters = list(inspect.signature(subclass.estimate).parameters)
        assert parameters == ["self", "kernel", "gpu"], subclass.__name__


def test_the_identity_audit_can_see_an_offender():
    """The control for the signature audit."""

    def price_for_rank(kernel, gpu, rank):  # pragma: no cover - inspected only
        raise NotImplementedError

    def price_for_worker(kernel, gpu, *, worker_id):  # pragma: no cover
        raise NotImplementedError

    def price_honestly(kernel, gpu):  # pragma: no cover - inspected only
        raise NotImplementedError

    assert study.identity_parameter_audit(
        {
            "offender.rank": price_for_rank,
            "offender.worker": price_for_worker,
            "clean": price_honestly,
        }
    ) == {"offender.rank": ["rank"], "offender.worker": ["worker_id"]}


def test_the_identity_audit_covers_every_pricing_entry_point():
    entries = set(study.pricing_entry_points())

    assert {
        "ComputeProvider.estimate",
        "RooflineProvider.estimate",
        "ProfileTableProvider.estimate",
        "TraceCalibratedGpuProvider.estimate",
        "_PinnedEnvelopeProvider.estimate",
        "SmSchedulerModel.estimate",
        "transformer.estimate_step_latency_ps",
    } <= entries


def test_the_audited_entry_points_do_not_depend_on_import_order():
    """A study's throwaway provider must not change what the audit covers.

    Loading a study that subclasses ``ComputeProvider`` used to enlarge the
    swept set, which made the audit's own reported coverage depend on what else
    the process had imported. The artifact byte lock is what found it.
    """

    before = sorted(study.pricing_entry_points())

    class _StudyLocalProvider(RooflineProvider):  # pragma: no cover - swept only
        pass

    after = sorted(study.pricing_entry_points())

    assert before == after
    assert _StudyLocalProvider not in study.first_party_providers()
    assert all(
        provider.__module__.startswith("simllm.")
        for provider in study.first_party_providers()
    )


def test_the_two_adapters_read_the_same_cost_geometry():
    vllm_dims, sglang_dims = study.adapter_dims()

    assert study.cost_geometry(vllm_dims) == study.cost_geometry(sglang_dims)
    assert vllm_dims.defaulted_fields == ()
    assert sglang_dims.defaulted_fields == ()


def test_the_two_adapters_price_one_step_to_the_same_picoseconds():
    vllm_dims, sglang_dims = study.adapter_dims()
    record = study.decode_record(2)

    vllm_ps = study._price(vllm_dims, record, 2)
    sglang_ps = study._price(sglang_dims, record, 2)

    assert vllm_ps == sglang_ps == study.ROOFLINE_PREDICTIONS["A1"][0]


def test_the_adapter_agreement_lock_sees_a_changed_geometry():
    """The negative control: a real geometry change must move the price."""

    from simllm.adapters.sglang.worker import model_dims_from_sglang

    record = study.decode_record(2)
    baseline = model_dims_from_sglang(study.sglang_model_config(), tp_size=1)
    sharded = model_dims_from_sglang(study.sglang_model_config(), tp_size=2)

    assert study._price(baseline, record, 2) != study._price(sharded, record, 2)


def test_the_two_readers_spell_the_optional_dtype_widths_differently():
    """A recorded representation difference that moves no picosecond (COMP-42).

    The vLLM reader resolves the weight and cache widths from the quantization
    and cache configs and stores explicit floats; the SGLang reader stores
    ``None`` and lets ``ModelDims`` fall back to the activation width. Both
    resolve to the same number through the properties the cost model reads. This
    test states the difference so a later normalization is a deliberate change
    rather than an accident.
    """

    vllm_dims, sglang_dims = study.adapter_dims()

    assert (vllm_dims.weight_dtype_bytes, vllm_dims.kv_dtype_bytes) == (2.0, 2.0)
    assert (sglang_dims.weight_dtype_bytes, sglang_dims.kv_dtype_bytes) == (None, None)
    assert vllm_dims.weight_element_bytes == sglang_dims.weight_element_bytes == 2.0
    assert vllm_dims.kv_element_bytes == sglang_dims.kv_element_bytes == 2.0
    assert vllm_dims != sglang_dims


# ---------------------------------------------------------------------------
# (c) Phase and token keying, at the exact frozen values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell_id", CELL_IDS)
def test_the_frozen_step_constants_hold(cell_id):
    expected_ps, expected_bound = study.ROOFLINE_PREDICTIONS[cell_id]

    estimate = _estimate(cell_id)

    assert (estimate.duration_ps, estimate.bound) == (expected_ps, expected_bound)


def test_the_phase_changes_the_bound_and_the_constant():
    decode = _estimate("A1")
    prefill = _estimate("A3")

    assert decode.bound == "memory"
    assert prefill.bound == "compute"
    assert decode.duration_ps != prefill.duration_ps


def test_more_decode_tokens_add_exactly_their_kv_read():
    two = _estimate("A1")
    four = _estimate("A2")

    kv_delta_bytes = (
        2 * 128 * study.NUM_LAYERS * study.NUM_KV_HEADS * study.HEAD_SIZE * study.DTYPE_BYTES
    )
    expected_delta = int(
        kv_delta_bytes / (study.MEM_BANDWIDTH * study.EFFICIENCY) * 1e12
    )

    assert four.duration_ps - two.duration_ps == expected_delta


def test_a_longer_prefill_chunk_grows_superlinearly():
    """The attention term is quadratic in the chunk, so 2x tokens is not 2x time."""

    short = _estimate("A3")
    long = _estimate("A4")

    assert long.duration_ps > 2 * short.duration_ps
    assert round(long.duration_ps / short.duration_ps, 4) == 2.6144


def test_the_ideal_host_profile_adds_exactly_zero():
    """CUDA graph against eager is a host term; it never touches kernel service."""

    ideal = HostInitiationModel.ideal()

    for cell_id in CELL_IDS:
        estimate = _estimate(cell_id)
        composed = ideal.represented_estimate(estimate, study.gpu_spec())
        assert composed.duration_ps == estimate.duration_ps
        assert composed.exposed_ps == 0
        assert composed.bound == estimate.bound


# ---------------------------------------------------------------------------
# (d) The memory-bound pin, on both paths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell_id", ["A1", "A2"])
def test_the_roofline_memory_bound_constant_is_the_hbm_bound(cell_id):
    factory, sampled = CELLS[cell_id]
    kernel = step_kernel(study.model_dims(), factory(), sampled)

    estimate = _estimate(cell_id)
    hbm_bound_ps = int(
        kernel.bytes_moved / (study.MEM_BANDWIDTH * study.EFFICIENCY) * 1e12
    )

    assert estimate.bound == "memory"
    assert estimate.duration_ps == hbm_bound_ps


@pytest.mark.parametrize(
    ("bandwidth", "expected_cycles"), [(64, 196), (32, 292)]
)
def test_the_mechanistic_constant_is_the_hbm_cursor_occupancy(bandwidth, expected_cycles):
    import math

    estimate = study.mechanistic_model(
        hbm_bandwidth=bandwidth, sm_count=1
    ).estimate(study.mechanistic_launch())

    instructions = study.MECHANISTIC_WARPS * study.MECHANISTIC_PER_WARP
    occupancy = instructions * math.ceil(
        study.MECHANISTIC_TRANSACTION_BYTES / bandwidth
    )

    assert estimate.duration_cycles == occupancy + study.TASK_MIX.HBM_LATENCY
    assert estimate.duration_cycles == expected_cycles
    assert estimate.duration_ps == expected_cycles * 1_000


def test_the_memory_bound_pin_ignores_sm_count():
    launch = study.mechanistic_launch()

    one_sm = study.mechanistic_model(hbm_bandwidth=64, sm_count=1).estimate(launch)
    two_sm = study.mechanistic_model(hbm_bandwidth=64, sm_count=2).estimate(launch)

    assert one_sm.duration_cycles == two_sm.duration_cycles == 196
    assert one_sm.hbm_transacted_bytes == two_sm.hbm_transacted_bytes


# ---------------------------------------------------------------------------
# The port taxonomy: nameable, and fail closed.
# ---------------------------------------------------------------------------


def _peer_port(protocol: GpuPortProtocol) -> GpuPortConfig:
    return GpuPortConfig(
        port_id=f"{protocol.value}-peer-store",
        protocol=protocol,
        role=GpuPortRole.PEER_LINK,
        direction=GpuPortDirection.EGRESS,
        capabilities=(GpuPortCapability.PEER_STORE_EGRESS,),
    )


@pytest.mark.parametrize(
    "protocol", [GpuPortProtocol.XGMI, GpuPortProtocol.UALINK]
)
def test_an_unprofiled_peer_protocol_fails_closed_and_names_its_owner(protocol):
    with pytest.raises(ValueError, match="COMP-35"):
        _peer_port(protocol)


def test_ualink_is_nameable_on_the_peer_link_role():
    """Naming it is the point: the diagnostic must be about the missing profile.

    A rejection that said "unknown protocol" would mean the taxonomy cannot
    describe a UALink topology at all, which is the opposite of what naming it
    is for.
    """

    from simllm.compute.gpu_device import _ROLE_PROTOCOLS

    assert GpuPortProtocol.UALINK.value == "ualink"
    assert GpuPortProtocol.UALINK in _ROLE_PROTOCOLS[GpuPortRole.PEER_LINK]
    assert GpuPortProtocol.UALINK not in _ROLE_PROTOCOLS[GpuPortRole.HOST_LINK]
    with pytest.raises(ValueError, match="no first-party or declared profile"):
        _peer_port(GpuPortProtocol.UALINK)


def test_the_xgmi_diagnostic_is_unchanged_by_the_ualink_addition():
    """COMP-34's accepted rejection text must survive the shared rule."""

    with pytest.raises(ValueError) as excinfo:
        _peer_port(GpuPortProtocol.XGMI)

    assert str(excinfo.value) == (
        "port 'xgmi-peer-store' declares the xGMI protocol, which has no "
        "first-party or declared profile in this repository; vendor port "
        "instantiation is COMP-35 and must supply a declared xGMI ceiling "
        "with its own provenance before an AMD cell may run"
    )


def test_the_profiled_peer_protocol_still_constructs():
    """Adding a protocol name must not disturb the one that has a mechanism."""

    port = _peer_port(GpuPortProtocol.NVLINK)

    assert port.protocol is GpuPortProtocol.NVLINK
    assert port.advertises(GpuPortCapability.PEER_STORE_EGRESS)
