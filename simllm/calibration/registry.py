"""Single-authority root resolution for offline suites and device releases."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path

SUITE_ROOT_ENV = "SIMLLM_CALIBRATION_SUITE_ROOT"
REGISTRY_ROOT_ENV = "SIMLLM_DEVICE_REGISTRY_ROOT"
PACKAGE_PROJECTION_FILE = ".simllm-package-projection.json"
PACKAGE_PROJECTION_SCHEMA = "simllm-immutable-root-projection-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROJECT_MARKERS = ("pyproject.toml", "simllm")


class RootResolutionError(ValueError):
    """Raised when exactly one valid root cannot be selected."""


class RootKind(str, Enum):
    """The two independent tracked root authorities."""

    SUITE = "suite"
    REGISTRY = "registry"


class RootSource(str, Enum):
    """How a root was selected."""

    EXPLICIT = "explicit"
    ENVIRONMENT = "environment"
    CHECKOUT = "checkout"
    PACKAGED = "packaged"


@dataclass(frozen=True)
class RootSelection:
    """One resolved root, never a merged view over several roots."""

    kind: RootKind
    source: RootSource
    root: Traversable
    manifest_sha256: str | None = None


@dataclass(frozen=True)
class _RootSpec:
    kind: RootKind
    environment_variable: str
    checkout_parts: tuple[str, ...]
    packaged_module: str


_SUITE_SPEC = _RootSpec(
    kind=RootKind.SUITE,
    environment_variable=SUITE_ROOT_ENV,
    checkout_parts=("offline", "calibration"),
    packaged_module="simllm._builtin_calibration",
)
_REGISTRY_SPEC = _RootSpec(
    kind=RootKind.REGISTRY,
    environment_variable=REGISTRY_ROOT_ENV,
    checkout_parts=("devices",),
    packaged_module="simllm._builtin_devices",
)


def resolve_suite_root(
    explicit_root: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
    packaged_root: Traversable | None = None,
) -> RootSelection:
    """Resolve the sole calibration-suite root using the frozen precedence."""

    return _resolve_root(
        _SUITE_SPEC,
        explicit_root=explicit_root,
        environ=environ,
        repository_root=repository_root,
        packaged_root=packaged_root,
    )


def resolve_registry_root(
    explicit_root: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
    packaged_root: Traversable | None = None,
) -> RootSelection:
    """Resolve the sole device-registry root using the frozen precedence."""

    return _resolve_root(
        _REGISTRY_SPEC,
        explicit_root=explicit_root,
        environ=environ,
        repository_root=repository_root,
        packaged_root=packaged_root,
    )


def _resolve_root(
    spec: _RootSpec,
    *,
    explicit_root: str | os.PathLike[str] | None,
    environ: Mapping[str, str] | None,
    repository_root: Path | None,
    packaged_root: Traversable | None,
) -> RootSelection:
    environment = os.environ if environ is None else environ
    explicit_path = _normalize_explicit_path(explicit_root)
    environment_path = _environment_path(environment, spec.environment_variable)

    if (
        explicit_path is not None
        and environment_path is not None
        and explicit_path != environment_path
    ):
        raise RootResolutionError(
            f"conflicting {spec.kind.value} roots: explicit path {explicit_path} "
            f"does not match {spec.environment_variable}={environment_path}"
        )

    if explicit_path is not None:
        return _filesystem_selection(spec, explicit_path, RootSource.EXPLICIT)
    if environment_path is not None:
        return _filesystem_selection(spec, environment_path, RootSource.ENVIRONMENT)

    project_root = _repository_root() if repository_root is None else repository_root
    if _is_repository_root(project_root):
        checkout_path = project_root.joinpath(*spec.checkout_parts)
        if checkout_path.is_dir():
            return _filesystem_selection(spec, checkout_path, RootSource.CHECKOUT)

    projection = packaged_root
    if projection is None:
        projection = _optional_package_root(spec.packaged_module)
    if projection is not None:
        return _packaged_selection(spec, projection)

    requirement = (
        f"--{spec.kind.value}-root or {spec.environment_variable}"
        if spec.kind is RootKind.REGISTRY
        else f"--suite-root or {spec.environment_variable}"
    )
    raise RootResolutionError(
        f"no {spec.kind.value} root is available; configure {requirement}"
    )


def _normalize_explicit_path(
    value: str | os.PathLike[str] | None,
) -> Path | None:
    if value is None:
        return None
    raw = os.fspath(value)
    if not raw.strip():
        raise RootResolutionError("an explicit root must not be blank")
    return Path(raw).expanduser().resolve(strict=False)


def _environment_path(
    environ: Mapping[str, str],
    variable: str,
) -> Path | None:
    raw = environ.get(variable)
    if raw is None or not raw.strip():
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        raise RootResolutionError(f"{variable} must be an absolute path")
    return path.resolve(strict=False)


def _filesystem_selection(
    spec: _RootSpec,
    path: Path,
    source: RootSource,
) -> RootSelection:
    if not path.exists():
        raise RootResolutionError(
            f"selected {spec.kind.value} root does not exist: {path}"
        )
    if not path.is_dir():
        raise RootResolutionError(
            f"selected {spec.kind.value} root is not a directory: {path}"
        )
    if spec.kind is RootKind.SUITE and not (path / "suites").is_dir():
        raise RootResolutionError(
            f"selected suite root has no suites directory: {path}"
        )
    return RootSelection(kind=spec.kind, source=source, root=path)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_repository_root(path: Path) -> bool:
    return (path / _PROJECT_MARKERS[0]).is_file() and (
        path / _PROJECT_MARKERS[1]
    ).is_dir()


def _optional_package_root(module_name: str) -> Traversable | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    return resources.files(module_name)


def _packaged_selection(spec: _RootSpec, root: Traversable) -> RootSelection:
    marker = root.joinpath(PACKAGE_PROJECTION_FILE)
    if not marker.is_file():
        raise RootResolutionError(
            f"packaged {spec.kind.value} root lacks {PACKAGE_PROJECTION_FILE}"
        )
    metadata = _read_projection_metadata(marker, spec.kind)
    manifest_path = metadata["manifest_path"]
    manifest = root.joinpath(manifest_path)
    if not manifest.is_file():
        raise RootResolutionError(
            f"packaged {spec.kind.value} manifest is missing: {manifest_path}"
        )
    actual_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source_digest = metadata["source_manifest_sha256"]
    projected_digest = metadata["projected_manifest_sha256"]
    if source_digest != projected_digest or actual_digest != projected_digest:
        raise RootResolutionError(
            f"packaged {spec.kind.value} manifest digest does not match its "
            "reviewed source manifest"
        )
    return RootSelection(
        kind=spec.kind,
        source=RootSource.PACKAGED,
        root=root,
        manifest_sha256=actual_digest,
    )


def _read_projection_metadata(marker: Traversable, kind: RootKind) -> dict[str, str]:
    from .canonical import strict_json_loads

    try:
        value = strict_json_loads(marker.read_bytes())
    except (OSError, UnicodeError, ValueError) as error:
        raise RootResolutionError(
            f"invalid packaged-root projection metadata: {error}"
        ) from error
    expected_keys = {
        "schema",
        "root_kind",
        "manifest_path",
        "source_manifest_sha256",
        "projected_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RootResolutionError("invalid packaged-root projection metadata fields")
    if value["schema"] != PACKAGE_PROJECTION_SCHEMA:
        raise RootResolutionError("unsupported packaged-root projection schema")
    if value["root_kind"] != kind.value:
        raise RootResolutionError("packaged-root projection kind does not match request")
    manifest_path = value["manifest_path"]
    if (
        not isinstance(manifest_path, str)
        or not manifest_path
        or manifest_path in {".", ".."}
        or "/" in manifest_path
        or "\\" in manifest_path
    ):
        raise RootResolutionError("packaged-root manifest path must be one file name")
    for field in ("source_manifest_sha256", "projected_manifest_sha256"):
        digest = value[field]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RootResolutionError(f"invalid {field} in packaged-root metadata")
    return value
