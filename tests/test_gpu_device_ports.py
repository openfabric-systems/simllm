"""Locks for the COMP-34 GPU device composition entry point.

Three things are locked here.

The off path: every accepted `gpu_task_mix`, `gpu_service_model` and
`mixed_makespan_v1` artifact must reproduce byte for byte when its harness is
driven through the composed device with default ports, and the composed device
must hand back the input architecture object itself. The negative control in
`test_the_byte_lock_sees_a_changed_port_ceiling` is what makes those assertions
mean something: with one declared ceiling at half the peer-link value, the same
regenerated artifact must differ.

The configuration boundary: a disabled port, a port asked for a capability it
does not advertise, an xGMI port, a peer-store port with no NVLink profile and a
copy direction the engine does not declare all reject during configuration
rather than at first use.

The enabled direction: the registered cells of
`examples/gpu_device_ports_v1/expectations.md`, so a later change that makes a
port ceiling inert or lets it rescope the wrong direction fails here.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.compute import (
    CopyDirection,
    CopyEngineServiceModel,
    GpuDevice,
    GpuDeviceConfig,
    GpuPortApplicability,
    GpuPortCapability,
    GpuPortCeiling,
    GpuPortCeilingProvenance,
    GpuPortConfig,
    GpuPortDirection,
    GpuPortProtocol,
    GpuPortRole,
    SmSchedulerModel,
    default_gpu_device_config,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = ROOT / "examples" / "gpu_device_ports_v1" / "run_study.py"
SPEC = importlib.util.spec_from_file_location("gpu_device_ports_v1_study", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
# Register before executing: the study defines dataclasses, and dataclasses
# resolves annotations through ``sys.modules`` while the class body runs.
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)

TASK_MIX_DIR = ROOT / "examples" / "gpu_task_mix"
SERVICE_MODEL_CSV = ROOT / "examples" / "gpu_service_model" / "results.csv"


# ---------------------------------------------------------------------------
# The off path, and the negative control that gives it teeth.
# ---------------------------------------------------------------------------


def test_default_ports_reproduce_the_task_mix_artifacts_byte_for_byte(tmp_path):
    exit_code, transcript = study._run_task_mix(tmp_path, ceilings=None)

    assert exit_code == 0
    for line in study.TASK_MIX_ACCEPTED_SUMMARY:
        assert line in transcript
    for name in study.TASK_MIX_ARTIFACTS:
        assert (tmp_path / name).read_bytes() == (TASK_MIX_DIR / name).read_bytes()


def test_the_byte_lock_sees_a_changed_port_ceiling(tmp_path):
    """The mutation-sensitive negative control for the lock above."""

    exit_code, _ = study._run_task_mix(tmp_path, ceilings={"nvlink-peer-store": 8.0})

    accepted = (TASK_MIX_DIR / "results.csv").read_bytes()
    assert (tmp_path / "results.csv").read_bytes() != accepted
    # The mutated run must also fail its own accepted rows. Changed bytes alone
    # could mean the harness stopped checking rather than that the mutation
    # reached the mechanism.
    assert exit_code != 0


def test_default_ports_reproduce_the_service_model_artifact_byte_for_byte(tmp_path):
    study._run_service_model(tmp_path / "results.csv")

    assert (tmp_path / "results.csv").read_bytes() == SERVICE_MODEL_CSV.read_bytes()


def test_the_service_model_lock_sees_a_changed_port_ceiling(tmp_path):
    """That artifact's own negative control, added after the freeze."""

    with pytest.raises(AssertionError):
        study._run_service_model(
            tmp_path / "results.csv", ceilings={"pcie-host-ingress-ce0": 32.0}
        )
    assert not (tmp_path / "results.csv").exists()


def test_default_ports_reproduce_the_mixed_makespan_record_byte_for_byte():
    bare = study._mixed_makespan_record(composed=False)
    composed = study._mixed_makespan_record(composed=True)

    assert composed == bare


def test_the_mixed_makespan_record_sees_a_changed_port_ceiling():
    """That record's own negative control, added after the freeze."""

    bare = study._mixed_makespan_record(composed=False)
    mutated = study._mixed_makespan_record(
        composed=True, ceilings={"nvlink-peer-store": 8.0}
    )

    assert mutated != bare


@pytest.mark.parametrize(
    "architecture",
    [
        study.TASK_MIX.architecture(),
        study.SERVICE_MODEL.architecture(),
        study.MIXED_MAKESPAN.architecture(),
        study.host_architecture(),
    ],
    ids=["task_mix", "service_model", "mixed_makespan", "host_fixture"],
)
def test_a_port_with_no_declared_ceiling_keeps_the_architecture_object(architecture):
    device = GpuDevice(default_gpu_device_config(architecture), architecture)

    assert device.architecture is architecture
    assert device.ports
    for port in device.ports:
        assert port.applicability is GpuPortApplicability.APPLICABLE
        assert port.ceiling is not None
        assert (
            port.ceiling.provenance is GpuPortCeilingProvenance.CALIBRATION_DERIVED
        )
        assert port.ceiling.bytes_per_cycle == study._mechanism_bandwidth(
            architecture, port
        )


def test_a_declared_ceiling_renames_the_derived_architecture():
    architecture = study.TASK_MIX.architecture()
    device = study._device(architecture, ceilings={"nvlink-peer-store": 8.0})

    assert device.architecture is not architecture
    assert device.architecture.profile_id.endswith("+nvlink-peer-store@8.0bpc")
    assert (
        device.architecture.calibration.target_architecture_profile_id
        == device.architecture.profile_id
    )
    assert device.architecture.calibration.nvlink is not None
    assert device.architecture.calibration.nvlink.bandwidth_bytes_per_cycle == 8.0
    # the base object is untouched: one authority, explicit projections
    assert architecture.calibration.nvlink is not None
    assert architecture.calibration.nvlink.bandwidth_bytes_per_cycle == 16
    assert "rescoped by simllm-gpu-device-v1" in (
        device.architecture.calibration.provenance.source
    )


def test_a_rescope_carries_live_provenance_and_an_honest_uncertainty():
    """A declared ceiling may not inherit a confidence or a date it did not earn."""

    base = study.TASK_MIX.architecture()
    measured = replace(
        base,
        calibration=replace(
            base.calibration,
            relative_uncertainty=0.02,
            provenance=replace(
                base.calibration.provenance,
                created="2020-01-01",
                references=("base-reference",),
            ),
        ),
    )
    config = GpuDeviceConfig(
        device_id="rescoped",
        ports=(
            _peer_store_port(
                capabilities=(
                    GpuPortCapability.PEER_STORE_EGRESS,
                    GpuPortCapability.CEILING_OVERRIDE,
                ),
                declared_ceiling=GpuPortCeiling(
                    bytes_per_cycle=8.0,
                    clock_hz=base.calibration.core_clock_hz,
                    provenance=GpuPortCeilingProvenance.VENDOR_NAMEPLATE,
                    source="vendor nameplate under test",
                    reference="declared-reference",
                    relative_uncertainty=0.30,
                    created="2026-08-17",
                ),
            ),
        ),
    )

    derived = GpuDevice(config, measured).architecture.calibration

    assert derived.relative_uncertainty == 0.30
    assert derived.provenance.created == "2026-08-17"
    assert derived.provenance.references == ("base-reference", "declared-reference")
    # the base calibration is untouched
    assert measured.calibration.relative_uncertainty == 0.02
    assert measured.calibration.provenance.created == "2020-01-01"


def test_an_unmeasured_rescope_may_not_keep_a_measured_uncertainty():
    base = study.TASK_MIX.architecture()
    measured = replace(
        base,
        calibration=replace(base.calibration, relative_uncertainty=0.02),
    )
    config = GpuDeviceConfig(
        device_id="inherited-confidence",
        ports=(
            _peer_store_port(
                capabilities=(
                    GpuPortCapability.PEER_STORE_EGRESS,
                    GpuPortCapability.CEILING_OVERRIDE,
                ),
                declared_ceiling=GpuPortCeiling(
                    bytes_per_cycle=8.0,
                    clock_hz=base.calibration.core_clock_hz,
                    provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
                    source="declared under test",
                    relative_uncertainty=0.01,
                    created="2026-08-17",
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="strictly wider uncertainty"):
        GpuDevice(config, measured)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("relative_uncertainty", None, "without a relative"),
        ("created", "", "without a created"),
    ],
)
def test_a_declared_ceiling_must_state_its_confidence_and_date(field, value, match):
    ceiling = GpuPortCeiling(
        bytes_per_cycle=8.0,
        clock_hz=1_000_000_000,
        provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
        source="test",
        relative_uncertainty=0.0,
        created="2026-08-17",
    )

    with pytest.raises(ValueError, match=match):
        _peer_store_port(
            capabilities=(
                GpuPortCapability.PEER_STORE_EGRESS,
                GpuPortCapability.CEILING_OVERRIDE,
            ),
            declared_ceiling=replace(ceiling, **{field: value}),
        )


def test_a_ceiling_read_from_the_mechanism_inherits_its_confidence_and_date():
    architecture = study.TASK_MIX.architecture()
    device = GpuDevice(default_gpu_device_config(architecture), architecture)
    ceiling = device.port("nvlink-peer-store").ceiling

    assert ceiling is not None
    assert ceiling.relative_uncertainty == architecture.calibration.relative_uncertainty
    assert ceiling.created == architecture.calibration.provenance.created


def test_the_composed_services_are_the_real_service_classes():
    architecture = study.host_architecture()
    device = study._device(architecture)

    assert isinstance(device.sm_scheduler_model(), SmSchedulerModel)
    assert isinstance(device.copy_engine_service("ce0"), CopyEngineServiceModel)


# ---------------------------------------------------------------------------
# Configuration-time rejection.
# ---------------------------------------------------------------------------


def _peer_store_port(**overrides) -> GpuPortConfig:
    fields = {
        "port_id": "nvlink-peer-store",
        "protocol": GpuPortProtocol.NVLINK,
        "role": GpuPortRole.PEER_LINK,
        "direction": GpuPortDirection.EGRESS,
        "capabilities": (GpuPortCapability.PEER_STORE_EGRESS,),
    }
    fields.update(overrides)
    return GpuPortConfig(**fields)


def test_a_disabled_port_may_not_carry_a_declared_ceiling():
    with pytest.raises(ValueError, match="disabled port"):
        study.host_port(
            port_id="ingress",
            direction=GpuPortDirection.INGRESS,
            copy_direction=CopyDirection.HOST_TO_DEVICE,
            enabled=False,
            ceiling=GpuPortCeiling(
                bytes_per_cycle=32.0,
                clock_hz=study.HOST_CLOCK_HZ,
                provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
                source="test",
                relative_uncertainty=0.0,
                created="2026-08-17",
            ),
        )


def test_a_declared_ceiling_requires_the_override_capability():
    with pytest.raises(ValueError, match="ceiling_override"):
        GpuPortConfig(
            port_id="nvlink-peer-store",
            protocol=GpuPortProtocol.NVLINK,
            role=GpuPortRole.PEER_LINK,
            direction=GpuPortDirection.EGRESS,
            capabilities=(GpuPortCapability.PEER_STORE_EGRESS,),
            declared_ceiling=GpuPortCeiling(
                bytes_per_cycle=8.0,
                clock_hz=1_000_000_000,
                provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
                source="test",
                relative_uncertainty=0.0,
                created="2026-08-17",
            ),
        )


def test_a_declared_ceiling_may_not_claim_calibration_provenance():
    with pytest.raises(ValueError, match="calibration-derived provenance"):
        _peer_store_port(
            capabilities=(
                GpuPortCapability.PEER_STORE_EGRESS,
                GpuPortCapability.CEILING_OVERRIDE,
            ),
            declared_ceiling=GpuPortCeiling(
                bytes_per_cycle=8.0,
                clock_hz=1_000_000_000,
                provenance=GpuPortCeilingProvenance.CALIBRATION_DERIVED,
                source="test",
                relative_uncertainty=0.0,
                created="2026-08-17",
            ),
        )


def test_a_declared_ceiling_must_match_the_mechanism_clock():
    architecture = study.TASK_MIX.architecture()
    config = GpuDeviceConfig(
        device_id="wrong-clock",
        ports=(
            _peer_store_port(
                capabilities=(
                    GpuPortCapability.PEER_STORE_EGRESS,
                    GpuPortCapability.CEILING_OVERRIDE,
                ),
                declared_ceiling=GpuPortCeiling(
                    bytes_per_cycle=8.0,
                    clock_hz=500_000_000,
                    provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
                    source="test",
                    relative_uncertainty=0.0,
                    created="2026-08-17",
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="clock"):
        GpuDevice(config, architecture)


def test_a_disabled_port_rejects_a_capability_request_at_configuration_time():
    architecture = study.host_architecture()
    device = GpuDevice(
        GpuDeviceConfig(
            device_id="disabled",
            ports=(
                study.host_port(
                    port_id="pcie-host-ingress",
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                    enabled=False,
                ),
            ),
        ),
        architecture,
    )

    with pytest.raises(ValueError, match="is disabled"):
        device.require_capability(
            "pcie-host-ingress", GpuPortCapability.COPY_ENGINE_TRANSFER
        )


def test_a_port_rejects_a_capability_it_does_not_advertise():
    device = study.host_device()

    with pytest.raises(ValueError, match="does not advertise"):
        device.require_capability(
            study.HOST_INGRESS_PORT, GpuPortCapability.PEER_STORE_EGRESS
        )


@pytest.mark.parametrize(
    "capability",
    [
        GpuPortCapability.ECN_MARKING,
        GpuPortCapability.PRIORITY_FLOW_CONTROL,
        GpuPortCapability.CONGESTION_NOTIFICATION,
    ],
)
def test_a_gpu_port_may_not_declare_a_transport_control_capability(capability):
    with pytest.raises(ValueError, match="BACK-48"):
        _peer_store_port(
            capabilities=(GpuPortCapability.PEER_STORE_EGRESS, capability)
        )


def test_an_xgmi_port_fails_closed_and_names_its_owner():
    with pytest.raises(ValueError, match="COMP-35"):
        _peer_store_port(port_id="xgmi-peer", protocol=GpuPortProtocol.XGMI)


def test_a_peer_store_port_needs_an_nvlink_profile():
    architecture = study.host_architecture()
    config = GpuDeviceConfig(device_id="no-nvlink", ports=(_peer_store_port(),))

    with pytest.raises(ValueError, match="no NVLink profile"):
        GpuDevice(config, architecture)


def test_an_undeclared_copy_direction_is_rejected_before_first_use():
    architecture = study.host_architecture()
    config = GpuDeviceConfig(
        device_id="undeclared-direction",
        ports=(
            GpuPortConfig(
                port_id="nvlink-peer-copy",
                protocol=GpuPortProtocol.NVLINK,
                role=GpuPortRole.PEER_LINK,
                direction=GpuPortDirection.EGRESS,
                capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
                copy_engine_id="ce0",
                copy_directions=(CopyDirection.PEER_TO_PEER,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not declare"):
        GpuDevice(config, architecture)


def test_an_unknown_copy_engine_is_rejected():
    architecture = study.host_architecture()
    config = GpuDeviceConfig(
        device_id="unknown-engine",
        ports=(
            study.host_port(
                port_id="pcie-host-ingress",
                direction=GpuPortDirection.INGRESS,
                copy_direction=CopyDirection.HOST_TO_DEVICE,
            ),
        ),
    )
    config = replace(
        config,
        ports=(replace(config.ports[0], copy_engine_id="ce-absent"),),
    )

    with pytest.raises(ValueError, match="ce-absent"):
        GpuDevice(config, architecture)


def test_one_mechanism_has_one_port_authority():
    with pytest.raises(ValueError, match="one port authority"):
        GpuDeviceConfig(
            device_id="two-authorities",
            ports=(
                _peer_store_port(port_id="a"),
                _peer_store_port(port_id="b"),
            ),
        )


def test_one_port_may_not_read_two_disagreeing_mechanism_ceilings():
    """The rejection the docs cite for the measured Grace C2C asymmetry."""

    architecture = study.host_architecture()
    asymmetric = replace(
        architecture,
        calibration=replace(
            architecture.calibration,
            copy_engines=(
                replace(
                    architecture.copy_engines[0],
                    direction_profiles=(
                        replace(
                            architecture.copy_engines[0].direction_profiles[0],
                            bandwidth_bytes_per_cycle=64.0,
                        ),
                        replace(
                            architecture.copy_engines[0].direction_profiles[1],
                            bandwidth_bytes_per_cycle=26.0,
                        ),
                    ),
                ),
            ),
        ),
    )
    config = GpuDeviceConfig(
        device_id="asymmetric-host",
        ports=(
            GpuPortConfig(
                port_id="pcie-host",
                protocol=GpuPortProtocol.PCIE,
                role=GpuPortRole.HOST_LINK,
                direction=GpuPortDirection.BIDIRECTIONAL,
                capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
                copy_engine_id="ce0",
                copy_directions=(
                    CopyDirection.HOST_TO_DEVICE,
                    CopyDirection.DEVICE_TO_HOST,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="ceilings disagree"):
        GpuDevice(config, asymmetric)


def test_a_bidirectional_port_must_name_every_direction_of_its_link():
    architecture = study.host_architecture()
    config = GpuDeviceConfig(
        device_id="half-bidirectional",
        ports=(
            GpuPortConfig(
                port_id="pcie-host",
                protocol=GpuPortProtocol.PCIE,
                role=GpuPortRole.HOST_LINK,
                direction=GpuPortDirection.BIDIRECTIONAL,
                capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
                copy_engine_id="ce0",
                copy_directions=(CopyDirection.HOST_TO_DEVICE,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not name every copy direction"):
        GpuDevice(config, architecture)


def test_a_direction_and_its_copy_direction_must_agree():
    with pytest.raises(ValueError, match="carried by"):
        GpuPortConfig(
            port_id="wrong-direction",
            protocol=GpuPortProtocol.PCIE,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.INGRESS,
            capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
            copy_engine_id="ce0",
            copy_directions=(CopyDirection.DEVICE_TO_HOST,),
        )


def test_an_unsupported_config_version_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        GpuDeviceConfig(device_id="future", version=99)
    with pytest.raises(ValueError, match="not supported"):
        _peer_store_port(version=99)


def test_an_enabled_port_must_declare_the_mechanism_it_wraps():
    with pytest.raises(ValueError, match="no mechanism capability"):
        GpuPortConfig(
            port_id="inert",
            protocol=GpuPortProtocol.PCIE,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.INGRESS,
            capabilities=(GpuPortCapability.CEILING_OVERRIDE,),
        )


# ---------------------------------------------------------------------------
# Applicability and inertness.
# ---------------------------------------------------------------------------


def test_a_disabled_port_keeps_its_interface_and_rescopes_nothing():
    architecture = study.host_architecture()
    device = GpuDevice(
        GpuDeviceConfig(
            device_id="disabled",
            ports=(
                study.host_port(
                    port_id=study.HOST_INGRESS_PORT,
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                    enabled=False,
                ),
            ),
        ),
        architecture,
    )
    port = device.port(study.HOST_INGRESS_PORT)

    assert port.applicability is GpuPortApplicability.NOT_APPLICABLE
    assert port.ceiling is None
    assert GpuPortCapability.COPY_ENGINE_TRANSFER in port.capabilities
    assert device.architecture is architecture
    # the mechanism behind the disabled declaration keeps its accepted timing
    assert study.dma_jct_ps(device, byte_count=4_096, to_device=True) == 84_000


def test_an_unknown_port_id_raises_a_key_error():
    device = study.host_device()

    with pytest.raises(KeyError, match="declares no port"):
        device.port("absent")


# ---------------------------------------------------------------------------
# The enabled direction: the registered cells.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("byte_count", "ceiling", "expected_ps"),
    [(key[0], key[1], value) for key, value in study.FROZEN_HOST_JCT_PS.items()],
)
def test_a_host_port_ceiling_moves_the_end_to_end_metric(
    byte_count, ceiling, expected_ps
):
    override = None if ceiling == study.HOST_BYTES_PER_CYCLE else ceiling
    device = study.host_device(ingress_ceiling=override)

    measured = study.dma_jct_ps(device, byte_count=byte_count, to_device=True)

    assert measured == expected_ps


@pytest.mark.parametrize(
    ("byte_count", "expected_ps"), sorted(study.FROZEN_HOST_ISOLATION_PS.items())
)
def test_a_host_port_ceiling_stays_inside_its_own_direction(byte_count, expected_ps):
    device = study.host_device(
        ingress_ceiling=study.ISOLATION_CEILING_BYTES_PER_CYCLE
    )

    measured = study.dma_jct_ps(device, byte_count=byte_count, to_device=False)

    assert measured == expected_ps


@pytest.mark.parametrize(
    ("chunk", "ceiling", "expected_cycles"),
    [(key[0], key[1], value) for key, value in study.FROZEN_PEER_CYCLES.items()],
)
def test_a_peer_port_ceiling_moves_the_egress_term(chunk, ceiling, expected_cycles):
    override = None if ceiling == 16.0 else ceiling

    measured = study.peer_egress_cycles(chunk_bytes=chunk, ceiling=override)

    assert measured == expected_cycles


def test_the_accepted_ring_row_holds_and_the_halved_ceiling_stays_in_band():
    low, high = study.FROZEN_RING_BAND_CYCLES

    assert study.ring_cycles(ceiling=None) == study.FROZEN_RING_BASELINE_CYCLES
    assert low <= study.ring_cycles(ceiling=8.0) <= high
