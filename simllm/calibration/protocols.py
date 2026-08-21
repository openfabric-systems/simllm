"""Backend-neutral protocols for offline evidence producers.

This module deliberately defines no concrete collector or simulator. Vendor
toolchains live behind these structural protocols and are never imported here.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["CalibrationCompiler", "HardwareCollector", "OfflineKernelSimulator"]

DoctorRequestT_contra = TypeVar("DoctorRequestT_contra", contravariant=True)
CaptureRequestT_contra = TypeVar("CaptureRequestT_contra", contravariant=True)
GraphT_contra = TypeVar("GraphT_contra", contravariant=True)
CapabilityReportT_co = TypeVar("CapabilityReportT_co", covariant=True)
EvidenceRecordT_co = TypeVar("EvidenceRecordT_co", covariant=True)

SimulationRequestT_contra = TypeVar(
    "SimulationRequestT_contra",
    contravariant=True,
)
BindingT_contra = TypeVar("BindingT_contra", contravariant=True)
TraceRefT_contra = TypeVar("TraceRefT_contra", contravariant=True)
CapabilityDecisionT_co = TypeVar("CapabilityDecisionT_co", covariant=True)
SimulatorObservationT_co = TypeVar("SimulatorObservationT_co", covariant=True)

EvidenceManifestT_contra = TypeVar("EvidenceManifestT_contra", contravariant=True)
FitRecordT = TypeVar("FitRecordT")
DeviceModelT = TypeVar("DeviceModelT")
ValidationRecordT_co = TypeVar("ValidationRecordT_co", covariant=True)


@runtime_checkable
class HardwareCollector(
    Protocol[
        DoctorRequestT_contra,
        CaptureRequestT_contra,
        GraphT_contra,
        CapabilityReportT_co,
        EvidenceRecordT_co,
    ]
):
    """Produce typed hardware capability and capture evidence offline."""

    def doctor(self, request: DoctorRequestT_contra, /) -> CapabilityReportT_co:
        """Describe whether the declared local environment can be captured."""

        ...

    def capture(
        self,
        request: CaptureRequestT_contra,
        graph: GraphT_contra,
        /,
    ) -> tuple[EvidenceRecordT_co, ...]:
        """Capture evidence for one exact graph without mutating that graph."""

        ...


@runtime_checkable
class OfflineKernelSimulator(
    Protocol[
        SimulationRequestT_contra,
        BindingT_contra,
        TraceRefT_contra,
        CapabilityDecisionT_co,
        SimulatorObservationT_co,
    ]
):
    """Query and run an optional offline kernel simulator."""

    def supports(
        self,
        request: SimulationRequestT_contra,
        binding: BindingT_contra,
        /,
    ) -> CapabilityDecisionT_co:
        """Return a typed support decision for one resolved implementation."""

        ...

    def replay(
        self,
        request: SimulationRequestT_contra,
        trace_ref: TraceRefT_contra,
        /,
    ) -> SimulatorObservationT_co:
        """Replay a content-addressed captured input and return one observation."""

        ...


@runtime_checkable
class CalibrationCompiler(
    Protocol[
        EvidenceManifestT_contra,
        FitRecordT,
        DeviceModelT,
        ValidationRecordT_co,
    ]
):
    """Compile validated offline evidence without target-specific policy."""

    def fit(self, evidence_manifest: EvidenceManifestT_contra, /) -> FitRecordT:
        """Fit only the manifest's immutable training partition."""

        ...

    def compile(self, fit: FitRecordT, /) -> DeviceModelT:
        """Compile a fit record into one compact device model."""

        ...

    def verify(
        self,
        evidence_manifest: EvidenceManifestT_contra,
        fit: FitRecordT,
        model: DeviceModelT,
        /,
    ) -> ValidationRecordT_co:
        """Score immutable held-out evidence and return a validation record."""

        ...
