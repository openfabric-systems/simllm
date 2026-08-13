"""Fixed per-step host initiation and launch-throughput profiles.

The default profile is the historical ideal model. It contributes exactly
zero and therefore preserves every accepted study that predates COMP-2.

The calibrated profiles describe one measured host and GPU pair. A step with
provider service ``C``, launch count ``N`` and per-launch throughput cost
``g`` is represented as ``max(C, N * g)``. The terms overlap because the host
can enqueue work while earlier kernels execute. Adding ``N * g`` to ``C``
would double count that overlap.

The Turing constants are deliberately device-bound sensitivity profiles, not
portable defaults. They reject every GPU key except ``gtx1660-ti-sm75``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from simllm.compute.provider import DurationEstimate, GpuSpec


@dataclass(frozen=True)
class HostStepEstimate:
    """One provider estimate after host launch-throughput composition."""

    duration_ps: int
    provider_duration_ps: int
    launch_floor_ps: int
    launch_floor_lower_ps: int
    launch_floor_upper_ps: int
    empirical_lower_ps: int
    empirical_upper_ps: int
    exposed_ps: int
    uncertainty_fraction: float
    bound: str


@dataclass(frozen=True)
class HostInitiationModel:
    """A fixed per-step host model with optional calibrated launch demand.

    The first two fields retain the original positional constructor. An
    arbitrary non-calibrated constant therefore keeps its historical additive
    behavior. New calibrated profiles must be built with the named factories;
    for those profiles ``initiation_delay_ps`` is the captured per-launch
    point and ``launch_floor_ps`` multiplies it by ``launch_count``. They use
    the overlap-aware ``max(provider, launch floor)`` composition.
    """

    initiation_delay_ps: int = 0
    profile: str = "ideal"
    launch_class: str = "none"
    launch_count: int = 0
    point_ps_per_launch: int = 0
    empirical_min_ps_per_launch: int = 0
    empirical_max_ps_per_launch: int = 0
    device_key: str | None = None
    device_model: str | None = None
    gpu_uuid: str | None = None
    host_cpu: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    source_study: str | None = None
    uncertainty_kind: str | None = None

    _TURING_DEVICE_KEY: ClassVar[str] = "gtx1660-ti-sm75"
    _TURING_DEVICE_MODEL: ClassVar[str] = "NVIDIA GeForce GTX 1660 Ti"
    _TURING_GPU_UUID: ClassVar[str] = (
        "GPU-a90a812a-41bf-4f2f-c96d-d83e6eae6bd0"
    )
    _TURING_HOST_CPU: ClassVar[str] = "AMD Ryzen 9 3950X 16-Core Processor"
    _TURING_DRIVER: ClassVar[str] = "550.90.07"
    _TURING_CUDA: ClassVar[str] = "12.4.99"
    _SOURCE_STUDY: ClassVar[str] = "examples/host_step_cost_v1"
    _UNCERTAINTY_KIND: ClassVar[str] = (
        "sample-limited empirical range, not a confidence interval"
    )

    def __post_init__(self) -> None:
        integer_fields = {
            "initiation_delay_ps": self.initiation_delay_ps,
            "launch_count": self.launch_count,
            "point_ps_per_launch": self.point_ps_per_launch,
            "empirical_min_ps_per_launch": self.empirical_min_ps_per_launch,
            "empirical_max_ps_per_launch": self.empirical_max_ps_per_launch,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

        if (
            self.profile == "ideal"
            and self.initiation_delay_ps > 0
            and self.launch_class == "none"
            and self.launch_count == 0
            and self.point_ps_per_launch == 0
            and self.empirical_min_ps_per_launch == 0
            and self.empirical_max_ps_per_launch == 0
        ):
            object.__setattr__(self, "profile", "legacy-fixed-step")

        if self.profile == "ideal":
            if any(integer_fields.values()) or self.launch_class != "none":
                raise ValueError("the ideal host profile must be exactly zero")
            if any(
                value is not None
                for value in (
                    self.device_key,
                    self.device_model,
                    self.gpu_uuid,
                    self.host_cpu,
                    self.driver_version,
                    self.cuda_version,
                    self.source_study,
                    self.uncertainty_kind,
                )
            ):
                raise ValueError("the ideal host profile carries no calibration provenance")
            return

        if self.profile in {"turing-cuda-graph", "turing-eager-host"}:
            expected = self._calibrated_values(self.profile)
            actual = (
                self.launch_class,
                self.point_ps_per_launch,
                self.empirical_min_ps_per_launch,
                self.empirical_max_ps_per_launch,
            )
            if actual != expected:
                raise ValueError(
                    f"{self.profile} constants must come from its named factory"
                )
            if self.launch_count <= 0:
                raise ValueError("a calibrated host profile needs a positive launch count")
            if self.initiation_delay_ps != self.point_ps_per_launch:
                raise ValueError("calibrated initiation delay must equal the per-launch point")
            expected_provenance = (
                self._TURING_DEVICE_KEY,
                self._TURING_DEVICE_MODEL,
                self._TURING_GPU_UUID,
                self._TURING_HOST_CPU,
                self._TURING_DRIVER,
                self._TURING_CUDA,
                self._SOURCE_STUDY,
                self._UNCERTAINTY_KIND,
            )
            actual_provenance = (
                self.device_key,
                self.device_model,
                self.gpu_uuid,
                self.host_cpu,
                self.driver_version,
                self.cuda_version,
                self.source_study,
                self.uncertainty_kind,
            )
            if actual_provenance != expected_provenance:
                raise ValueError(
                    f"{self.profile} provenance must come from its named factory"
                )

    @staticmethod
    def _calibrated_values(profile: str) -> tuple[str, int, int, int]:
        if profile == "turing-cuda-graph":
            return ("cuda-graph-node", 809_306, 624_665, 809_306)
        if profile == "turing-eager-host":
            return ("eager-host-bound", 2_364_255, 2_327_730, 2_544_074)
        raise ValueError(f"unknown calibrated host profile {profile!r}")

    @classmethod
    def ideal(cls) -> HostInitiationModel:
        """Return the exact historical zero-cost profile."""

        return cls()

    @classmethod
    def _turing_profile(cls, profile: str, launch_count: int) -> HostInitiationModel:
        if isinstance(launch_count, bool) or not isinstance(launch_count, int):
            raise TypeError("launch_count must be an integer")
        if launch_count <= 0:
            raise ValueError("launch_count must be positive")
        launch_class, point, empirical_min, empirical_max = cls._calibrated_values(
            profile
        )
        return cls(
            initiation_delay_ps=point,
            profile=profile,
            launch_class=launch_class,
            launch_count=launch_count,
            point_ps_per_launch=point,
            empirical_min_ps_per_launch=empirical_min,
            empirical_max_ps_per_launch=empirical_max,
            device_key=cls._TURING_DEVICE_KEY,
            device_model=cls._TURING_DEVICE_MODEL,
            gpu_uuid=cls._TURING_GPU_UUID,
            host_cpu=cls._TURING_HOST_CPU,
            driver_version=cls._TURING_DRIVER,
            cuda_version=cls._TURING_CUDA,
            source_study=cls._SOURCE_STUDY,
            uncertainty_kind=cls._UNCERTAINTY_KIND,
        )

    @classmethod
    def turing_cuda_graph(cls, launch_count: int) -> HostInitiationModel:
        """Build the accepted GTX 1660 Ti CUDA-graph launch profile."""

        return cls._turing_profile("turing-cuda-graph", launch_count)

    @classmethod
    def turing_eager_host(cls, launch_count: int) -> HostInitiationModel:
        """Build the accepted GTX 1660 Ti eager host-launch profile."""

        return cls._turing_profile("turing-eager-host", launch_count)

    @property
    def is_ideal(self) -> bool:
        """Whether this is the exact identity profile."""

        return self.profile == "ideal"

    @property
    def is_calibrated(self) -> bool:
        """Whether this is one of the device-bound accepted profiles."""

        return self.profile in {"turing-cuda-graph", "turing-eager-host"}

    @property
    def lower_ps_per_launch(self) -> int:
        """Lower endpoint of the empirical per-launch range."""

        return self.empirical_min_ps_per_launch

    @property
    def device_name(self) -> str | None:
        """Captured display name of the calibrated GPU."""

        return self.device_model

    @property
    def upper_ps_per_launch(self) -> int:
        """Upper endpoint of the empirical per-launch range."""

        return self.empirical_max_ps_per_launch

    @property
    def launch_floor_ps(self) -> int:
        """Point fixed-step launch floor, ``launch_count * g``."""

        if self.is_calibrated:
            return self.launch_count * self.point_ps_per_launch
        return self.initiation_delay_ps

    def delay_ps(self) -> int:
        """Return the legacy scalar or calibrated per-launch point value."""

        return self.initiation_delay_ps

    def validate_device(self, gpu: GpuSpec) -> None:
        """Fail closed when a calibrated constant meets another GPU."""

        if not isinstance(gpu, GpuSpec):
            raise TypeError("gpu must be a GpuSpec")
        if self.is_calibrated and gpu.name != self.device_key:
            raise ValueError(
                f"host profile {self.profile!r} is calibrated only for "
                f"GpuSpec.name={self.device_key!r}, not {gpu.name!r}"
            )

    @staticmethod
    def _provider_bounds(estimate: DurationEstimate) -> tuple[int, int]:
        lower = max(0, math.floor(estimate.duration_ps * (1.0 - estimate.uncertainty)))
        upper = math.ceil(estimate.duration_ps * (1.0 + estimate.uncertainty))
        return lower, upper

    def represented_estimate(
        self,
        provider_estimate: DurationEstimate,
        gpu: GpuSpec,
    ) -> HostStepEstimate:
        """Compose provider service with this profile exactly once."""

        if not isinstance(provider_estimate, DurationEstimate):
            raise TypeError("provider_estimate must be a DurationEstimate")
        if provider_estimate.duration_ps < 0:
            raise ValueError("provider duration must be nonnegative")
        if provider_estimate.uncertainty < 0:
            raise ValueError("provider uncertainty must be nonnegative")
        self.validate_device(gpu)
        provider_lower, provider_upper = self._provider_bounds(provider_estimate)

        if self.is_ideal:
            return HostStepEstimate(
                duration_ps=provider_estimate.duration_ps,
                provider_duration_ps=provider_estimate.duration_ps,
                launch_floor_ps=0,
                launch_floor_lower_ps=0,
                launch_floor_upper_ps=0,
                empirical_lower_ps=provider_lower,
                empirical_upper_ps=provider_upper,
                exposed_ps=0,
                uncertainty_fraction=provider_estimate.uncertainty,
                bound=provider_estimate.bound,
            )

        if self.is_calibrated:
            launch_floor = self.launch_floor_ps
            launch_lower = self.launch_count * self.empirical_min_ps_per_launch
            launch_upper = self.launch_count * self.empirical_max_ps_per_launch
            duration = max(provider_estimate.duration_ps, launch_floor)
            lower = max(provider_lower, launch_lower)
            upper = max(provider_upper, launch_upper)
        else:
            launch_floor = self.initiation_delay_ps
            launch_lower = launch_floor
            launch_upper = launch_floor
            duration = provider_estimate.duration_ps + launch_floor
            lower = provider_lower + launch_lower
            upper = provider_upper + launch_upper

        uncertainty = 0.0
        if duration > 0:
            uncertainty = max(duration - lower, upper - duration) / duration
        bound = (
            provider_estimate.bound
            if duration == provider_estimate.duration_ps
            else "host-initiation"
        )
        return HostStepEstimate(
            duration_ps=duration,
            provider_duration_ps=provider_estimate.duration_ps,
            launch_floor_ps=launch_floor,
            launch_floor_lower_ps=launch_lower,
            launch_floor_upper_ps=launch_upper,
            empirical_lower_ps=lower,
            empirical_upper_ps=upper,
            exposed_ps=duration - provider_estimate.duration_ps,
            uncertainty_fraction=uncertainty,
            bound=bound,
        )
