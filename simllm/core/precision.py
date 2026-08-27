"""Validated run-wide precision selection and provenance.

The public configuration names the eight seams in the repository fidelity
matrix. Operational objects such as providers, placement manifests and backend
profiles remain the mechanism selectors. Their owning modules resolve those
legacy spellings to these levels and reject an explicit disagreement before
execution.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from simllm.core._wire import _enum_value, _fields, _object, _string

PRECISION_CONFIG_SCHEMA = "simllm-precision-config-v1"
RUN_PROVENANCE_SCHEMA = "simllm-run-provenance-v1"

PRECISION_SEAMS = (
    "workload",
    "request_outcome",
    "framework",
    "compute",
    "dependency",
    "locality",
    "network",
    "rnic_hardware",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class WorkloadLevel(str, enum.Enum):
    """How request arrivals and lengths enter the run."""

    FIXED_TRACE = "fixed-trace"
    POISSON_ARRIVALS = "poisson-arrivals"


class RequestOutcomeLevel(str, enum.Enum):
    """How token identity, output length and stop reason are fixed."""

    FABRICATED = "fabricated"
    PREPLAY_ORACLE = "preplay-oracle"
    FRAMEWORK_CPU_ORACLE = "framework-cpu-oracle"


class FrameworkLevel(str, enum.Enum):
    """Where framework scheduling crosses into SimLLM."""

    RECORDED_STEPS = "recorded-steps"
    EXECUTOR_RPC = "executor-rpc"
    MODEL_RUNNER = "model-runner"


class ComputeLevel(str, enum.Enum):
    """How GPU service duration is selected."""

    FIXED = "fixed"
    ROOFLINE = "roofline"
    PROFILE_TABLE = "profile-table"


class DependencyLevel(str, enum.Enum):
    """How operation dependencies are obtained."""

    SERIAL = "serial"
    OBSERVED_FRAMEWORK_SCHEDULE = "observed-framework-schedule"


class LocalityLevel(str, enum.Enum):
    """How intra-node and fabric traffic are distinguished."""

    ALL_REMOTE = "all-remote"
    ANALYTIC_NVLINK = "analytic-nvlink"


class NetworkLevel(str, enum.Enum):
    """How network completion is modeled."""

    RNIC_NN_FLUID = "rnic-nn-fluid"
    ANALYTIC_COMPOSITION = "analytic-composition"
    PACKET_LEVEL = "packet-level"


class RnicHardwareLevel(str, enum.Enum):
    """Whether the run uses the timing-neutral or structural RNIC path."""

    TIMING_NEUTRAL_BYPASS = "timing-neutral-bypass"
    COMPOSED_NATIVE = "composed-native"


_LEVEL_TYPES = {
    "workload": WorkloadLevel,
    "request_outcome": RequestOutcomeLevel,
    "framework": FrameworkLevel,
    "compute": ComputeLevel,
    "dependency": DependencyLevel,
    "locality": LocalityLevel,
    "network": NetworkLevel,
    "rnic_hardware": RnicHardwareLevel,
}

#: class attribute a :class:`~simllm.compute.ComputeProvider` may declare to
#: name the level it implements. A provider that omits it stays unresolved
#: rather than being assumed to be any particular level.
COMPUTE_LEVEL_ATTRIBUTE = "precision_compute_level"

#: backend profile spelling to the network level it selects
PROFILE_NETWORK_LEVELS = {
    "rnic-nn-fluid": NetworkLevel.RNIC_NN_FLUID,
    "analytic-composition": NetworkLevel.ANALYTIC_COMPOSITION,
    "rnic-nn": NetworkLevel.PACKET_LEVEL,
    "rnic-cn": NetworkLevel.PACKET_LEVEL,
    "dcqcn": NetworkLevel.PACKET_LEVEL,
}

#: authority-mode spelling to the RNIC hardware level it selects
AUTHORITY_MODE_RNIC_LEVELS = {
    "bypass": RnicHardwareLevel.TIMING_NEUTRAL_BYPASS,
    "structural": RnicHardwareLevel.COMPOSED_NATIVE,
}


@dataclass(frozen=True, kw_only=True)
class PrecisionConfig:
    """One validated precision level for every fidelity seam."""

    workload: WorkloadLevel
    request_outcome: RequestOutcomeLevel
    framework: FrameworkLevel
    compute: ComputeLevel
    dependency: DependencyLevel
    locality: LocalityLevel
    network: NetworkLevel
    rnic_hardware: RnicHardwareLevel

    def __post_init__(self) -> None:
        for name, expected_type in _LEVEL_TYPES.items():
            value = getattr(self, name)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"precision.{name}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        if (
            self.network is NetworkLevel.RNIC_NN_FLUID
            and self.rnic_hardware is RnicHardwareLevel.COMPOSED_NATIVE
        ):
            raise ValueError(
                "precision.rnic_hardware='composed-native' is incompatible with "
                "precision.network='rnic-nn-fluid'; select "
                "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
            )
        if (
            self.network is NetworkLevel.ANALYTIC_COMPOSITION
            and self.rnic_hardware is RnicHardwareLevel.COMPOSED_NATIVE
        ):
            raise ValueError(
                "precision.rnic_hardware='composed-native' is incompatible with "
                "precision.network='analytic-composition'; select "
                "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
            )

    @classmethod
    def compatibility(cls) -> PrecisionConfig:
        """Return the byte-locked compatibility level at every seam."""

        return cls(
            workload=WorkloadLevel.FIXED_TRACE,
            request_outcome=RequestOutcomeLevel.FABRICATED,
            framework=FrameworkLevel.RECORDED_STEPS,
            compute=ComputeLevel.FIXED,
            dependency=DependencyLevel.SERIAL,
            locality=LocalityLevel.ALL_REMOTE,
            network=NetworkLevel.RNIC_NN_FLUID,
            rnic_hardware=RnicHardwareLevel.TIMING_NEUTRAL_BYPASS,
        )

    def stamp(self, *, source_schema: str, source_sha256: str) -> RunProvenance:
        """Bind this resolved selection to one source artifact identity."""

        return RunProvenance(
            source_schema=source_schema,
            source_sha256=source_sha256,
            precision=self,
        )


def _validate_precision(value: PrecisionConfig) -> None:
    if not isinstance(value, PrecisionConfig):
        raise TypeError("precision: expected PrecisionConfig")
    value.__post_init__()


def precision_config_to_json(value: PrecisionConfig) -> dict[str, str]:
    """Return the strict JSON object for one complete precision selection."""

    _validate_precision(value)
    return {
        "schema": PRECISION_CONFIG_SCHEMA,
        **{name: getattr(value, name).value for name in PRECISION_SEAMS},
    }


def precision_config_from_json(value: object) -> PrecisionConfig:
    """Parse and validate one strict precision configuration object."""

    payload = _object(value, "precision")
    _fields(payload, "precision", required={"schema", *PRECISION_SEAMS})
    schema = _string(payload["schema"], "precision.schema")
    if schema != PRECISION_CONFIG_SCHEMA:
        raise ValueError(
            "precision.schema: unsupported schema "
            f"{schema!r}; expected {PRECISION_CONFIG_SCHEMA!r}"
        )
    arguments = {
        name: _enum_value(level_type, payload[name], f"precision.{name}")
        for name, level_type in _LEVEL_TYPES.items()
    }
    return PrecisionConfig(**arguments)


def precision_config_to_bytes(value: PrecisionConfig) -> bytes:
    """Return canonical UTF-8 bytes used by the precision stamp hash."""

    return _canonical_json(precision_config_to_json(value))


def precision_config_sha256(value: PrecisionConfig) -> str:
    """Return the canonical content identity of a precision configuration."""

    return hashlib.sha256(precision_config_to_bytes(value)).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RunProvenance:
    """One source schema and hash stamped with its resolved precision."""

    source_schema: str
    source_sha256: str
    precision: PrecisionConfig
    schema: str = RUN_PROVENANCE_SCHEMA
    precision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != RUN_PROVENANCE_SCHEMA:
            raise ValueError(
                "run_provenance.schema: unsupported schema "
                f"{self.schema!r}; expected {RUN_PROVENANCE_SCHEMA!r}"
            )
        source_schema = _string(
            self.source_schema,
            "run_provenance.source_schema",
        )
        if source_schema != self.source_schema:
            raise AssertionError("validated source schema changed unexpectedly")
        source_sha256 = _string(
            self.source_sha256,
            "run_provenance.source_sha256",
        )
        if _SHA256_RE.fullmatch(source_sha256) is None:
            raise ValueError(
                "run_provenance.source_sha256: expected 64 lowercase "
                "hexadecimal digits"
            )
        _validate_precision(self.precision)
        object.__setattr__(
            self,
            "precision_sha256",
            precision_config_sha256(self.precision),
        )

    @classmethod
    def from_source_bytes(
        cls,
        *,
        source_schema: str,
        source_bytes: bytes,
        precision: PrecisionConfig,
    ) -> RunProvenance:
        """Hash source bytes and bind their schema to a precision selection."""

        if not isinstance(source_bytes, bytes):
            raise TypeError("source_bytes: expected bytes")
        return cls(
            source_schema=source_schema,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            precision=precision,
        )


def run_provenance_to_json(value: RunProvenance) -> dict[str, Any]:
    """Return the strict JSON object for one stamped run provenance."""

    if not isinstance(value, RunProvenance):
        raise TypeError("run_provenance: expected RunProvenance")
    value.__post_init__()
    return {
        "schema": value.schema,
        "source_schema": value.source_schema,
        "source_sha256": value.source_sha256,
        "precision": precision_config_to_json(value.precision),
        "precision_sha256": value.precision_sha256,
    }


def run_provenance_from_json(value: object) -> RunProvenance:
    """Parse and validate one strict run provenance object."""

    payload = _object(value, "run_provenance")
    _fields(
        payload,
        "run_provenance",
        required={
            "schema",
            "source_schema",
            "source_sha256",
            "precision",
            "precision_sha256",
        },
    )
    schema = _string(payload["schema"], "run_provenance.schema")
    precision = precision_config_from_json(payload["precision"])
    record = RunProvenance(
        schema=schema,
        source_schema=_string(
            payload["source_schema"],
            "run_provenance.source_schema",
        ),
        source_sha256=_string(
            payload["source_sha256"],
            "run_provenance.source_sha256",
        ),
        precision=precision,
    )
    observed_hash = _string(
        payload["precision_sha256"],
        "run_provenance.precision_sha256",
    )
    if _SHA256_RE.fullmatch(observed_hash) is None:
        raise ValueError(
            "run_provenance.precision_sha256: expected 64 lowercase "
            "hexadecimal digits"
        )
    if observed_hash != record.precision_sha256:
        raise ValueError("run_provenance.precision_sha256: digest mismatch")
    return record


def run_provenance_to_bytes(value: RunProvenance) -> bytes:
    """Return canonical newline-terminated UTF-8 provenance bytes."""

    return _canonical_json(run_provenance_to_json(value)) + b"\n"


def run_provenance_from_bytes(value: bytes) -> RunProvenance:
    """Parse one UTF-8 JSON run provenance payload."""

    if not isinstance(value, bytes):
        raise TypeError("run_provenance_bytes: expected bytes")
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("run_provenance_bytes: invalid UTF-8 JSON") from exc
    return run_provenance_from_json(payload)


def write_run_provenance(
    value: RunProvenance,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one canonical provenance object with protective creation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    with target.open(mode) as stream:
        stream.write(run_provenance_to_bytes(value))
    return target


def read_run_provenance(path: str | Path) -> RunProvenance:
    """Read one strict run provenance object."""

    return run_provenance_from_bytes(Path(path).read_bytes())


def network_level_for_profile(profile: str) -> NetworkLevel:
    """Return the network level the backend profile spelling selects."""

    name = _string(profile, "profile")
    try:
        return PROFILE_NETWORK_LEVELS[name]
    except KeyError:
        choices = ", ".join(sorted(PROFILE_NETWORK_LEVELS))
        raise ValueError(
            f"profile: {name!r} selects no known network level; expected one "
            f"of {choices}"
        ) from None


def rnic_hardware_level_for_authority_mode(mode: object) -> RnicHardwareLevel:
    """Return the RNIC hardware level the WQE authority mode selects."""

    name = _string(mode, "authority_mode")
    try:
        return AUTHORITY_MODE_RNIC_LEVELS[name]
    except KeyError:
        choices = ", ".join(sorted(AUTHORITY_MODE_RNIC_LEVELS))
        raise ValueError(
            f"authority_mode: {name!r} selects no known RNIC hardware level; "
            f"expected one of {choices}"
        ) from None


def locality_level_for_placement(placement_manifest: object | None) -> LocalityLevel:
    """Return the locality level the placement-manifest spelling selects."""

    if placement_manifest is None:
        return LocalityLevel.ALL_REMOTE
    return LocalityLevel.ANALYTIC_NVLINK


def dependency_level_for_observations(observations: object | None) -> DependencyLevel:
    """Return the dependency level the observation-supply spelling selects."""

    if observations is None:
        return DependencyLevel.SERIAL
    return DependencyLevel.OBSERVED_FRAMEWORK_SCHEDULE


def compute_level_for_provider(provider: object) -> ComputeLevel | None:
    """Return the level a compute provider declares, or ``None`` if it does not.

    Providers name their own level through the
    :data:`COMPUTE_LEVEL_ATTRIBUTE` class attribute. A caller-defined provider
    that declares nothing resolves to ``None``: the surface records that the
    legacy spelling constrains nothing here rather than guessing a level.
    """

    declared = getattr(provider, COMPUTE_LEVEL_ATTRIBUTE, None)
    if declared is None:
        return None
    if isinstance(declared, ComputeLevel):
        return declared
    return _enum_value(
        ComputeLevel,
        declared,
        f"provider.{COMPUTE_LEVEL_ATTRIBUTE}",
    )


def check_precision_selection(
    precision: PrecisionConfig | None = None,
    *,
    workload: WorkloadLevel | None = None,
    request_outcome: RequestOutcomeLevel | None = None,
    framework: FrameworkLevel | None = None,
    compute: ComputeLevel | None = None,
    dependency: DependencyLevel | None = None,
    locality: LocalityLevel | None = None,
    network: NetworkLevel | None = None,
    rnic_hardware: RnicHardwareLevel | None = None,
    selection_source: str = "legacy selection",
) -> dict[str, Any]:
    """Return the observed seam levels, refusing an explicit disagreement.

    A component observes only the seams its own spellings select. This
    function reports exactly those and never invents a level for a seam the
    caller could not see, so a partial view can neither claim nor be refused
    on a seam it does not own. When an explicit configuration is supplied,
    every observed level must equal the configured one; the conflict names
    the component through ``selection_source``.
    """

    source = _string(selection_source, "selection_source")
    selected = {
        "workload": workload,
        "request_outcome": request_outcome,
        "framework": framework,
        "compute": compute,
        "dependency": dependency,
        "locality": locality,
        "network": network,
        "rnic_hardware": rnic_hardware,
    }
    for name, value in selected.items():
        if value is not None and not isinstance(value, _LEVEL_TYPES[name]):
            raise TypeError(
                f"{source}.{name}: expected {_LEVEL_TYPES[name].__name__}, "
                f"got {type(value).__name__}"
            )
    observed = {name: value for name, value in selected.items() if value is not None}
    if precision is None:
        return observed
    _validate_precision(precision)
    for name, selected_level in observed.items():
        configured = getattr(precision, name)
        if configured is not selected_level:
            raise ValueError(
                f"precision.{name}={configured.value!r} conflicts with {source}, "
                f"which selects {selected_level.value!r}"
            )
    return observed


def resolve_precision_config(
    precision: PrecisionConfig | None = None,
    *,
    workload: WorkloadLevel | None = None,
    request_outcome: RequestOutcomeLevel | None = None,
    framework: FrameworkLevel | None = None,
    compute: ComputeLevel | None = None,
    dependency: DependencyLevel | None = None,
    locality: LocalityLevel | None = None,
    network: NetworkLevel | None = None,
    rnic_hardware: RnicHardwareLevel | None = None,
    selection_source: str = "legacy selection",
) -> PrecisionConfig:
    """Compose one complete run configuration from the observed levels.

    This is the run-level composer, not the per-component check: the caller
    asserts a complete run, so a seam no observation names takes its
    compatibility level and the result is validated as a whole. A completion
    that pairs incompatible levels is refused rather than degraded, which is
    why a component that observes a single seam calls
    :func:`check_precision_selection` instead. When an explicit configuration
    is supplied it is returned unchanged after every observed level agrees.
    """

    observed = check_precision_selection(
        precision,
        workload=workload,
        request_outcome=request_outcome,
        framework=framework,
        compute=compute,
        dependency=dependency,
        locality=locality,
        network=network,
        rnic_hardware=rnic_hardware,
        selection_source=selection_source,
    )
    if precision is None:
        return replace(PrecisionConfig.compatibility(), **observed)
    return precision


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "AUTHORITY_MODE_RNIC_LEVELS",
    "COMPUTE_LEVEL_ATTRIBUTE",
    "PRECISION_CONFIG_SCHEMA",
    "PRECISION_SEAMS",
    "PROFILE_NETWORK_LEVELS",
    "RUN_PROVENANCE_SCHEMA",
    "ComputeLevel",
    "DependencyLevel",
    "FrameworkLevel",
    "LocalityLevel",
    "NetworkLevel",
    "PrecisionConfig",
    "RequestOutcomeLevel",
    "RnicHardwareLevel",
    "RunProvenance",
    "WorkloadLevel",
    "check_precision_selection",
    "compute_level_for_provider",
    "dependency_level_for_observations",
    "locality_level_for_placement",
    "network_level_for_profile",
    "precision_config_from_json",
    "precision_config_sha256",
    "precision_config_to_bytes",
    "precision_config_to_json",
    "read_run_provenance",
    "resolve_precision_config",
    "rnic_hardware_level_for_authority_mode",
    "run_provenance_from_bytes",
    "run_provenance_from_json",
    "run_provenance_to_bytes",
    "run_provenance_to_json",
    "write_run_provenance",
]
