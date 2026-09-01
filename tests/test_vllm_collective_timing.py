from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.adapters.vllm.collective_capture as live_capture
from simllm.core import (
    COLLECTIVE_SERVICE_SCHEMA,
    STEP_SCHEMA,
    CollectiveServiceCapture,
    CollectiveServiceEnvironment,
    CollectiveServiceInvocation,
    StepRecord,
    step_record_from_json,
    step_record_to_json,
)
from simllm.traffic import (
    CollectiveFloorCalibration,
    CollectiveFloorEnvironmentMismatchError,
    CollectiveFloorRegime,
    CollectiveFloorSourceIdentity,
    compare_collective_service_to_floor,
)

_STUDY_PATH = (
    Path(__file__).parents[1] / "examples" / "vllm_collective_timing_v1" / "run_study.py"
)
_STUDY_SPEC = importlib.util.spec_from_file_location("vllm48_run_study", _STUDY_PATH)
assert _STUDY_SPEC is not None and _STUDY_SPEC.loader is not None
run_study = importlib.util.module_from_spec(_STUDY_SPEC)
_STUDY_SPEC.loader.exec_module(run_study)


class _FakeDtype:
    def __str__(self) -> str:
        return "torch.float16"


class _Tensor:
    def __init__(self, shape=(2, 4), *, device_type="cpu") -> None:
        self.shape = shape
        self.dtype = _FakeDtype()
        self.device = SimpleNamespace(type=device_type)

    def numel(self) -> int:
        product = 1
        for extent in self.shape:
            product *= extent
        return product

    def element_size(self) -> int:
        return 2


def _environment(*, system="capture-host", backend="gloo", timer="host-monotonic-ns"):
    return CollectiveServiceEnvironment(
        system=system,
        backend=backend,
        device_type="cpu",
        framework="vllm",
        framework_version="0.27.1+cpu",
        timer=timer,
    )


def _invocation(**overrides):
    values = {
        "sequence": 0,
        "kind": "all_reduce",
        "payload_bytes": 16,
        "world_size": 2,
        "dtype": "float16",
        "element_width_bytes": 2,
        "tensor_shape": (2, 4),
        "group_tag": "tp:0",
        "service_ps": 200,
        "layer_index": 3,
        "layer_name": "model.layers.3",
    }
    values.update(overrides)
    return CollectiveServiceInvocation(**values)


def _calibration(*, system="floor-host", backend="nccl"):
    source = CollectiveFloorSourceIdentity(
        artifact_sha256="0" * 64,
        tool="fixture",
        aiconfigurator_version="1",
        aiconfigurator_core_version="1",
        system=system,
        backend=backend,
        database_version="1",
        row_version="1",
        duplicate_resolution="none",
    )
    regime = CollectiveFloorRegime(
        dtype="half",
        operation="all_reduce",
        ranks=2,
        regime_index=0,
        lower_bytes=1,
        upper_bytes=1024,
        floor_ps=Fraction(100),
        slope_ps_per_byte=Fraction(1),
        training_cell_ids=("a", "b"),
    )
    return CollectiveFloorCalibration(
        calibration_id="fixture-floor",
        source=source,
        fitted_byte_range=(1, 1024),
        regimes=(regime,),
    )


def test_absent_collective_field_preserves_old_canonical_bytes():
    old_bytes = (
        b'{"finished_request_ids":[],"preempted_request_ids":[],"scheduled":[],'
        b'"schema":"atlahs-closed-loop-step-v1","step_index":4,'
        b'"virtual_time_ps":9}'
    )
    loaded = step_record_from_json(json.loads(old_bytes))
    assert loaded.collective_service is None
    assert (
        json.dumps(step_record_to_json(loaded), sort_keys=True, separators=(",", ":")).encode()
        == old_bytes
    )
    assert "collective_service" not in step_record_to_json(
        StepRecord(step_index=0, virtual_time_ps=0)
    )


def test_new_collective_field_round_trips_canonical_bytes():
    record = StepRecord(
        step_index=4,
        virtual_time_ps=9,
        collective_service=CollectiveServiceCapture(
            environment=_environment(), invocations=(_invocation(),)
        ),
    )
    encoded = json.dumps(
        step_record_to_json(record), sort_keys=True, separators=(",", ":")
    ).encode()
    loaded = step_record_from_json(json.loads(encoded))
    assert loaded == record
    assert loaded.collective_service.schema == COLLECTIVE_SERVICE_SCHEMA
    assert step_record_to_json(loaded)["schema"] == STEP_SCHEMA
    assert (
        json.dumps(step_record_to_json(loaded), sort_keys=True, separators=(",", ":")).encode()
        == encoded
    )


def test_cpu_seam_records_payload_group_layer_and_positive_service(monkeypatch):
    monkeypatch.setenv(live_capture.COLLECTIVE_CAPTURE_SYSTEM_ENV, "cpu-host")
    monkeypatch.setattr(live_capture.importlib.metadata, "version", lambda _name: "0.27.1+cpu")
    coordinator = SimpleNamespace(
        world_size=2,
        rank=0,
        torch_distributed_backend="gloo",
        unique_name="tp:0",
    )
    session = live_capture._CaptureSession(pending=[])
    session_token = live_capture._active_session.set(session)
    layer_token = live_capture._active_layers.set(((7, "model.layers.7"),))
    calls = []

    def collective(_coordinator, tensor):
        calls.append(tensor)
        return tensor

    try:
        tensor = _Tensor()
        assert (
            live_capture._timed_collective_call(coordinator, "all_reduce", collective, tensor)
            is tensor
        )
    finally:
        live_capture._active_layers.reset(layer_token)
        live_capture._active_session.reset(session_token)

    resolved = session.resolve()
    assert calls == [tensor]
    assert resolved.environment == _environment(system="cpu-host")
    assert len(resolved.invocations) == 1
    invocation = resolved.invocations[0]
    assert invocation.kind == "all_reduce"
    assert invocation.payload_bytes == 16
    assert invocation.tensor_shape == (2, 4)
    assert invocation.world_size == 2
    assert invocation.group_tag == "tp:0"
    assert invocation.layer_index == 7
    assert invocation.layer_name == "model.layers.7"
    assert invocation.service_ps > 0


def test_cuda_seam_uses_events_around_the_collective(monkeypatch):
    monkeypatch.setenv(live_capture.COLLECTIVE_CAPTURE_SYSTEM_ENV, "gpu-host")
    monkeypatch.setattr(live_capture.importlib.metadata, "version", lambda _name: "0.27.1")
    actions = []

    class Event:
        def __init__(self, name):
            self.name = name

        def record(self):
            actions.append(f"record:{self.name}")

        def synchronize(self):
            actions.append(f"sync:{self.name}")

        def elapsed_time(self, other):
            actions.append(f"elapsed:{self.name}:{other.name}")
            return 0.25

    monkeypatch.setattr(live_capture, "_cuda_events", lambda: (Event("start"), Event("end")))
    coordinator = SimpleNamespace(
        world_size=2,
        rank=0,
        torch_distributed_backend="nccl",
        unique_name="tp:0",
    )
    session = live_capture._CaptureSession(pending=[])
    token = live_capture._active_session.set(session)

    def collective(_coordinator, tensor):
        actions.append("collective")
        return tensor

    try:
        live_capture._timed_collective_call(
            coordinator, "all_gather", collective, _Tensor(device_type="cuda")
        )
    finally:
        live_capture._active_session.reset(token)
    assert actions == ["record:start", "collective", "record:end"]
    actions.append("step-returned")
    capture = session.resolve()
    assert actions == [
        "record:start",
        "collective",
        "record:end",
        "step-returned",
        "sync:end",
        "elapsed:start:end",
    ]
    assert capture.environment.timer == "cuda-event"
    assert capture.invocations[0].service_ps == 250_000_000


def test_cuda_execute_defers_resolution_beyond_the_wrapped_step(monkeypatch):
    monkeypatch.setenv(live_capture.COLLECTIVE_CAPTURE_SYSTEM_ENV, "gpu-host")
    monkeypatch.setattr(live_capture.importlib.metadata, "version", lambda _name: "0.27.1")
    actions = []

    class Event:
        def __init__(self, name):
            self.name = name

        def record(self):
            actions.append(f"record:{self.name}")

        def synchronize(self):
            actions.append(f"sync:{self.name}")

        def elapsed_time(self, other):
            actions.append(f"elapsed:{self.name}:{other.name}")
            return 0.25

    monkeypatch.setattr(live_capture, "_cuda_events", lambda: (Event("start"), Event("end")))
    monkeypatch.setattr(
        live_capture,
        "translate_scheduler_output",
        lambda *_args, **_kwargs: SimpleNamespace(
            record=StepRecord(step_index=0, virtual_time_ps=0)
        ),
    )
    monkeypatch.setattr(live_capture, "_step_index", 0)
    deferred = []

    def submit(record, session):
        actions.append("defer:resolution")
        deferred.append((record, session))

    monkeypatch.setattr(live_capture, "_submit_cuda_resolution", submit)
    monkeypatch.setattr(
        live_capture,
        "_append_record",
        lambda _record: actions.append("append:record"),
    )
    coordinator = SimpleNamespace(
        world_size=2,
        rank=0,
        torch_distributed_backend="nccl",
        unique_name="tp:0",
    )

    def execute(_runner, _scheduler_output):
        live_capture._timed_collective_call(
            coordinator,
            "all_gather",
            lambda _coordinator, tensor: actions.append("collective") or tensor,
            _Tensor(device_type="cuda"),
        )
        actions.append("execute:return")
        return "result"

    assert live_capture._capture_execute(execute, object(), object()) == "result"
    actions.append("wrapper:return")
    assert actions == [
        "record:start",
        "collective",
        "record:end",
        "execute:return",
        "defer:resolution",
        "wrapper:return",
    ]
    live_capture._resolve_and_append(*deferred[0])
    assert actions[-3:] == ["sync:end", "elapsed:start:end", "append:record"]


def test_non_driver_and_single_rank_calls_are_not_recorded():
    session = live_capture._CaptureSession(pending=[])
    token = live_capture._active_session.set(session)

    def collective(_coordinator, tensor):
        return tensor

    try:
        tensor = _Tensor()
        for rank, world_size in ((1, 2), (0, 1)):
            coordinator = SimpleNamespace(
                world_size=world_size,
                rank=rank,
                torch_distributed_backend="gloo",
                unique_name="tp:0",
            )
            live_capture._timed_collective_call(coordinator, "all_reduce", collective, tensor)
    finally:
        live_capture._active_session.reset(token)
    assert session.pending == []


def test_comparator_refuses_cross_environment_by_default():
    with pytest.raises(
        CollectiveFloorEnvironmentMismatchError,
        match="acknowledge_cross_environment=True",
    ):
        compare_collective_service_to_floor(
            invocation=_invocation(),
            environment=_environment(),
            calibration=_calibration(),
            floor_dtype="half",
        )


def test_comparator_stamps_every_acknowledged_cross_environment_result():
    result = compare_collective_service_to_floor(
        invocation=_invocation(),
        environment=_environment(),
        calibration=_calibration(),
        floor_dtype="half",
        acknowledge_cross_environment=True,
    )
    assert result.estimate.requested_operation == "all_reduce"
    assert result.estimate.message_bytes == 16
    assert result.estimate.requested_ranks == 2
    assert result.estimate.completion_ps == 116
    assert result.residual_ps == 84
    assert result.observed_to_floor_ratio == Fraction(50, 29)
    assert result.cross_environment_acknowledged
    assert result.as_dict()["cross_environment_acknowledged"] is True


def test_same_environment_comparison_is_unstamped():
    result = compare_collective_service_to_floor(
        invocation=_invocation(),
        environment=_environment(system="floor-host", backend="nccl"),
        calibration=_calibration(),
        floor_dtype="half",
    )
    assert not result.cross_environment_acknowledged
    assert not result.transferred_at_use_acknowledged


def test_request_identity_amendment_requires_exact_vllm_suffix():
    assert run_study._internal_request_id_matches("vllm48-short", "vllm48-short-0123abcd")
    assert not run_study._internal_request_id_matches("vllm48-short", "vllm48-short")
    assert not run_study._internal_request_id_matches(
        "vllm48-short", "vllm48-short-0123abcd-extra"
    )
    assert not run_study._internal_request_id_matches("vllm48-short", "other-0123abcd")


def test_metadata_mutations_flip_after_masking_only_refuted_kind_cell():
    expected_step = [
        {
            "sequence": index,
            "kind": "gather" if index == 49 else "all_reduce",
            "payload_bytes": 16,
        }
        for index in range(50)
    ]
    observed_step = deepcopy(expected_step)
    observed_step[49]["kind"] = "all_gather"
    expected_runs = [[expected_step]]
    observed_runs = [[observed_step]]
    shape = [{"step_index": 0, "invocations": observed_step}]
    raw_rows = [
        step_record_to_json(
            StepRecord(
                step_index=0,
                virtual_time_ps=0,
                collective_service=CollectiveServiceCapture(
                    environment=_environment(),
                    invocations=(_invocation(),),
                ),
            )
        )
    ]
    transitions = run_study._mutation_controls(
        expected_runs,
        observed_runs,
        [shape, deepcopy(shape)],
        raw_rows,
        True,
        {"cross_environment_acknowledged": True},
    )
    assert all(control["baseline_pass"] for control in transitions.values())
    assert not any(control["mutant_pass"] for control in transitions.values())
    assert all(control["pass_to_fail"] for control in transitions.values())
