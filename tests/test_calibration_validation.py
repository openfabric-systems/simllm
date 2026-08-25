"""Fail-closed tests for the hardware-independent calibration validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import simllm.calibration.canonical as canonical_module
from simllm.calibration.canonical import canonical_bytes, canonical_sha256
from simllm.calibration.cli import main
from simllm.calibration.validation import (
    CalibrationValidationError,
    ValidationResult,
    validate_path,
    validate_typed_record,
)
from simllm.compute.device_model import ShapeAxis, ShapeSchema


def _shape_schema_obj() -> dict[str, object]:
    return ShapeSchema(
        shape_schema_id="attention-v1",
        axes=(
            ShapeAxis(axis_id="batch", unit="requests", minimum=1, maximum=64),
            ShapeAxis(axis_id="tokens", unit="tokens", minimum=1, maximum=131072),
        ),
    ).to_obj()


def _device_model_obj() -> dict[str, object]:
    registry = {
        "schema": "simllm-device-resource-registry-v1",
        "device_kind_id": "test-device",
        "active_axis_ids": ["hbm-bytes"],
        "axes": [
            {
                "axis_id": "hbm-bytes",
                "axis_class": "throughput",
                "service_scope": "device-internal",
                "base_unit": "bytes",
                "clock_domain_id": None,
                "capacity_source_id": "test-capacity",
                "rate": {"numerator": 1, "denominator": 1},
                "residency_capacity": None,
                "exclusive_capacity": None,
            }
        ],
    }
    registry_sha256 = canonical_sha256(registry)
    digest = "1" * 64
    shape_schema = _shape_schema_obj()
    return {
        "schema": "simllm-device-model-v1",
        "device_model_id": "test-model-v1",
        "device_kind_id": "test-device",
        "acceptance_status": "candidate",
        "target_basis": "target-silicon",
        "device_identity_sha256": digest,
        "operating_envelope_sha256": digest,
        "support_envelope_sha256": digest,
        "evidence_manifest_sha256": digest,
        "fit_sha256": digest,
        "expectations_commit": "a" * 40,
        "dispatch_signature_sha256s": [digest],
        "shape_schemas": [shape_schema],
        "implementation_selector_sha256": digest,
        "collective_stage_selector_sha256": None,
        "resource_registry": registry,
        "interaction_contract": {
            "interaction_law": "independent-resource-v1",
            "interaction_terms": [],
        },
        "host_initiation_profile_sha256": None,
        "service_entries": [
            {
                "service_entry_id": "entry-1",
                "entry": {
                    "implementation_id": "kernel-1",
                    "shape_vector": {
                        "shape_schema_id": shape_schema["shape_schema_id"],
                        "values": [1, 128],
                    },
                    "epochs": [
                        {
                            "resource_vector": {
                                "registry_sha256": registry_sha256,
                                "device_kind_id": "test-device",
                                "values": [4096],
                                "known": [True],
                            },
                            "fixed_floor_ps": 10,
                        }
                    ],
                },
            }
        ],
        "service_entry_evidence": [
            {
                "service_entry_id": "entry-1",
                "source_selection": "silicon",
                "source_record_sha256s": [digest],
                "residual_record_sha256": digest,
                "support_envelope_sha256": digest,
                "operating_envelope_sha256": digest,
                "isolated_duration_ps": 100,
                "uncertainty_bound": {"numerator": 1, "denominator": 10},
            }
        ],
        "scalar_profile_table_sha256": None,
        "gpu_spec_sha256": None,
        "gpu_architecture_profile_sha256": None,
        "gpu_device_config_sha256": None,
        "validation_record_sha256": digest,
        "validation_summary_sha256": digest,
        "acceptance_bars_sha256": digest,
        "model_limits": {
            "max_shape_schemas": 1,
            "max_shape_axes_per_schema": 2,
            "max_resource_axes": 1,
            "max_service_entries": 1,
            "max_epochs_per_entry": 1,
            "max_resident_entries": 8,
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value))


def test_validate_path_accepts_one_canonical_typed_record(tmp_path: Path) -> None:
    value = _shape_schema_obj()
    path = tmp_path / "shape.json"
    _write(path, value)

    result = validate_path(path)

    assert result == ValidationResult(
        record_schema="simllm-shape-schema-v1",
        record_sha256=canonical_sha256(value),
        size_bytes=len(canonical_bytes(value)),
    )
    assert result.to_obj() == {
        "valid": True,
        "record_schema": "simllm-shape-schema-v1",
        "record_sha256": canonical_sha256(value),
        "size_bytes": len(canonical_bytes(value)),
    }


def test_validate_typed_record_returns_the_strict_component() -> None:
    typed = validate_typed_record(_shape_schema_obj())
    assert isinstance(typed, ShapeSchema)
    assert typed.to_obj() == _shape_schema_obj()


def test_validate_typed_record_checks_inline_registry_content_identity() -> None:
    value = _device_model_obj()
    typed = validate_typed_record(value)
    assert typed.to_obj() == value

    entry = value["service_entries"][0]["entry"]  # type: ignore[index]
    vector = entry["epochs"][0]["resource_vector"]  # type: ignore[index]
    vector["registry_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(CalibrationValidationError, match="selected registry"):
        validate_typed_record(value)


def test_validate_path_accepts_complete_compact_device_model(tmp_path: Path) -> None:
    value = _device_model_obj()
    path = tmp_path / "device-model.json"
    _write(path, value)
    result = validate_path(path)
    assert result.record_schema == "simllm-device-model-v1"
    assert result.record_sha256 == canonical_sha256(value)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema":"simllm-shape-schema-v1","schema":"duplicate"}',
        b'{"schema":"simllm-shape-schema-v1"}\n',
        b'{"schema":"simllm-shape-schema-v1","value":1.5}',
        b"PK\x03\x04not-an-admitted-archive",
    ),
)
def test_validate_path_rejects_noncanonical_or_malicious_bytes(
    tmp_path: Path,
    raw: bytes,
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(CalibrationValidationError, match="invalid calibration record"):
        validate_path(path)


def test_validate_path_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    _write(path, {"schema": "third-party-unknown-v1", "value": 1})
    with pytest.raises(CalibrationValidationError, match="unsupported calibration"):
        validate_path(path)


def test_validate_path_reports_unsupported_unicode_runtime_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(
        b'{"axes":[],"schema":"simllm-shape-schema-v1",'
        b'"shape_schema_id":"caf\xc3\xa9"}'
    )

    def reject_runtime() -> None:
        raise RuntimeError("unsupported canonical Unicode runtime")

    monkeypatch.setattr(canonical_module, "assert_canonical_runtime", reject_runtime)
    with pytest.raises(
        CalibrationValidationError,
        match="unsupported canonical Unicode runtime",
    ):
        validate_path(path)


def test_validate_path_rejects_unknown_typed_fields(tmp_path: Path) -> None:
    value = _shape_schema_obj()
    value["unexpected"] = True
    path = tmp_path / "record.json"
    _write(path, value)
    with pytest.raises(CalibrationValidationError, match="unknown fields"):
        validate_path(path)


def test_validate_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(CalibrationValidationError, match="regular record file"):
        validate_path(tmp_path)


def test_validate_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CalibrationValidationError, match="does not exist"):
        validate_path(tmp_path / "missing.json")


def test_validate_path_rejects_direct_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write(target, _shape_schema_obj())
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(CalibrationValidationError, match="symbolic link"):
        validate_path(alias)


def test_validate_path_rejects_parent_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    _write(target / "record.json", _shape_schema_obj())
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(CalibrationValidationError, match="symbolic link"):
        validate_path(alias / "record.json")


def test_validate_path_rejects_size_over_limit(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    _write(path, _shape_schema_obj())
    with pytest.raises(CalibrationValidationError, match="limit is 8"):
        validate_path(path, max_record_bytes=8)


@pytest.mark.parametrize("limit", (0, -1, True, 1.5))
def test_validate_path_rejects_invalid_limit(tmp_path: Path, limit: object) -> None:
    path = tmp_path / "record.json"
    _write(path, _shape_schema_obj())
    with pytest.raises(ValueError, match="positive integer"):
        validate_path(path, max_record_bytes=limit)  # type: ignore[arg-type]


def test_cli_projection_never_exposes_the_local_path(tmp_path: Path) -> None:
    value = _shape_schema_obj()
    path = tmp_path / "record.json"
    _write(path, value)
    encoded = json.dumps(validate_path(path).to_obj(), sort_keys=True)
    assert str(tmp_path) not in encoded


def test_validate_command_uses_the_real_hardware_independent_validator(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = _shape_schema_obj()
    path = tmp_path / "record.json"
    _write(path, value)

    assert main(["validate", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "record_schema": "simllm-shape-schema-v1",
        "record_sha256": canonical_sha256(value),
        "size_bytes": len(canonical_bytes(value)),
        "valid": True,
    }


def test_validate_command_loads_no_hardware_or_simulator_runtime(tmp_path: Path) -> None:
    value = _shape_schema_obj()
    path = tmp_path / "record.json"
    _write(path, value)
    source = (
        "import sys; from simllm.calibration.cli import main; "
        f"assert main(['validate', {str(path)!r}]) == 0; "
        "forbidden=('torch','cupy','cuda','rocm','rocprofiler','accel_sim'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in forbidden)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True
