from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simllm.calibration.registry import (
    PACKAGE_PROJECTION_FILE,
    PACKAGE_PROJECTION_SCHEMA,
    REGISTRY_ROOT_ENV,
    SUITE_ROOT_ENV,
    RootKind,
    RootResolutionError,
    RootSource,
    resolve_registry_root,
    resolve_suite_root,
)


def _repository(tmp_path: Path, *, suites: bool = False, devices: bool = False) -> Path:
    root = tmp_path / "checkout"
    (root / "simllm").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    if suites:
        (root / "offline" / "calibration" / "suites").mkdir(parents=True)
    if devices:
        (root / "devices").mkdir()
    return root


def _suite_root(tmp_path: Path, name: str = "suite-root") -> Path:
    root = tmp_path / name
    (root / "suites").mkdir(parents=True)
    return root


def _projection(
    tmp_path: Path,
    kind: RootKind,
    *,
    source_digest: str | None = None,
    projected_digest: str | None = None,
    metadata_update: dict[str, object] | None = None,
) -> Path:
    root = tmp_path / f"packaged-{kind.value}"
    root.mkdir()
    manifest = b'{"schema":"test-manifest-v1"}\n'
    (root / "manifest.json").write_bytes(manifest)
    actual_digest = hashlib.sha256(manifest).hexdigest()
    metadata: dict[str, object] = {
        "schema": PACKAGE_PROJECTION_SCHEMA,
        "root_kind": kind.value,
        "manifest_path": "manifest.json",
        "source_manifest_sha256": source_digest or actual_digest,
        "projected_manifest_sha256": projected_digest or actual_digest,
    }
    if metadata_update:
        metadata.update(metadata_update)
    (root / PACKAGE_PROJECTION_FILE).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return root


def test_explicit_suite_root_is_selected() -> None:
    selection = resolve_suite_root(
        "offline/calibration",
        environ={},
        repository_root=Path("missing-checkout"),
    )

    assert selection.kind is RootKind.SUITE
    assert selection.source is RootSource.EXPLICIT
    assert selection.root == Path("offline/calibration").resolve()
    assert selection.manifest_sha256 is None


def test_equal_explicit_and_environment_roots_are_one_authority(tmp_path: Path) -> None:
    root = _suite_root(tmp_path)

    selection = resolve_suite_root(
        root,
        environ={SUITE_ROOT_ENV: str(root)},
        repository_root=tmp_path / "missing",
    )

    assert selection.source is RootSource.EXPLICIT
    assert selection.root == root.resolve()


def test_conflicting_explicit_and_environment_roots_reject(tmp_path: Path) -> None:
    explicit = _suite_root(tmp_path, "explicit")
    environment = _suite_root(tmp_path, "environment")

    with pytest.raises(RootResolutionError, match="conflicting suite roots"):
        resolve_suite_root(
            explicit,
            environ={SUITE_ROOT_ENV: str(environment)},
            repository_root=tmp_path / "missing",
        )


def test_missing_explicit_root_does_not_fall_through_to_checkout(tmp_path: Path) -> None:
    checkout = _repository(tmp_path, suites=True)

    with pytest.raises(RootResolutionError, match="does not exist"):
        resolve_suite_root(
            tmp_path / "missing",
            environ={},
            repository_root=checkout,
        )


def test_missing_environment_root_does_not_fall_through_to_checkout(
    tmp_path: Path,
) -> None:
    checkout = _repository(tmp_path, devices=True)

    with pytest.raises(RootResolutionError, match="does not exist"):
        resolve_registry_root(
            environ={REGISTRY_ROOT_ENV: str(tmp_path / "missing")},
            repository_root=checkout,
        )


def test_invalid_explicit_suite_root_does_not_fall_through(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    checkout = _repository(tmp_path, suites=True)

    with pytest.raises(RootResolutionError, match="has no suites directory"):
        resolve_suite_root(invalid, environ={}, repository_root=checkout)


def test_environment_root_must_be_absolute(tmp_path: Path) -> None:
    checkout = _repository(tmp_path, suites=True)

    with pytest.raises(RootResolutionError, match="must be an absolute path"):
        resolve_suite_root(
            environ={SUITE_ROOT_ENV: "offline/calibration"},
            repository_root=checkout,
        )


def test_checkout_roots_are_derived_from_the_repository_root(tmp_path: Path) -> None:
    checkout = _repository(tmp_path, suites=True, devices=True)

    suite = resolve_suite_root(environ={}, repository_root=checkout)
    registry = resolve_registry_root(environ={}, repository_root=checkout)

    assert suite == type(suite)(
        kind=RootKind.SUITE,
        source=RootSource.CHECKOUT,
        root=checkout / "offline" / "calibration",
    )
    assert registry == type(registry)(
        kind=RootKind.REGISTRY,
        source=RootSource.CHECKOUT,
        root=checkout / "devices",
    )


def test_registry_environment_root_uses_its_own_variable(tmp_path: Path) -> None:
    registry = tmp_path / "devices"
    registry.mkdir()

    selection = resolve_registry_root(
        environ={REGISTRY_ROOT_ENV: str(registry)},
        repository_root=tmp_path / "missing",
    )

    assert selection.source is RootSource.ENVIRONMENT
    assert selection.root == registry.resolve()


@pytest.mark.parametrize("kind", [RootKind.SUITE, RootKind.REGISTRY])
def test_digest_checked_packaged_fallback_is_selected(
    tmp_path: Path,
    kind: RootKind,
) -> None:
    packaged = _projection(tmp_path, kind)
    resolver = resolve_suite_root if kind is RootKind.SUITE else resolve_registry_root

    selection = resolver(
        environ={},
        repository_root=tmp_path / "missing",
        packaged_root=packaged,
    )

    assert selection.source is RootSource.PACKAGED
    assert selection.root == packaged
    assert selection.manifest_sha256 == hashlib.sha256(
        (packaged / "manifest.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("source_digest", "projected_digest"),
    [
        ("0" * 64, None),
        (None, "f" * 64),
        ("0" * 64, "f" * 64),
        ("0" * 64, "0" * 64),
    ],
)
def test_packaged_manifest_digest_mismatch_rejects(
    tmp_path: Path,
    source_digest: str | None,
    projected_digest: str | None,
) -> None:
    packaged = _projection(
        tmp_path,
        RootKind.REGISTRY,
        source_digest=source_digest,
        projected_digest=projected_digest,
    )

    with pytest.raises(RootResolutionError, match="does not match"):
        resolve_registry_root(
            environ={},
            repository_root=tmp_path / "missing",
            packaged_root=packaged,
        )


@pytest.mark.parametrize(
    "metadata_update",
    [
        {"schema": "unknown"},
        {"root_kind": "suite"},
        {"manifest_path": "../manifest.json"},
        {"source_manifest_sha256": "ABC"},
        {"unknown": True},
    ],
)
def test_invalid_packaged_projection_metadata_rejects(
    tmp_path: Path,
    metadata_update: dict[str, object],
) -> None:
    packaged = _projection(
        tmp_path,
        RootKind.REGISTRY,
        metadata_update=metadata_update,
    )

    with pytest.raises(RootResolutionError):
        resolve_registry_root(
            environ={},
            repository_root=tmp_path / "missing",
            packaged_root=packaged,
        )


def test_duplicate_packaged_projection_field_rejects(tmp_path: Path) -> None:
    packaged = _projection(tmp_path, RootKind.REGISTRY)
    marker = packaged / PACKAGE_PROJECTION_FILE
    original = marker.read_text(encoding="utf-8")
    marker.write_text(
        original.replace("{", '{"schema":"duplicate",', 1),
        encoding="utf-8",
    )

    with pytest.raises(RootResolutionError, match="duplicate"):
        resolve_registry_root(
            environ={},
            repository_root=tmp_path / "missing",
            packaged_root=packaged,
        )


def test_checkout_precedes_and_does_not_merge_packaged_content(tmp_path: Path) -> None:
    checkout = _repository(tmp_path, devices=True)
    invalid_packaged = tmp_path / "invalid-packaged"
    invalid_packaged.mkdir()

    selection = resolve_registry_root(
        environ={},
        repository_root=checkout,
        packaged_root=invalid_packaged,
    )

    assert selection.source is RootSource.CHECKOUT
    assert selection.root == checkout / "devices"


def test_no_available_root_fails_with_actionable_configuration(tmp_path: Path) -> None:
    with pytest.raises(
        RootResolutionError,
        match=REGISTRY_ROOT_ENV,
    ):
        resolve_registry_root(
            environ={},
            repository_root=tmp_path / "missing",
            packaged_root=None,
        )
